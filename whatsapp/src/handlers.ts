/**
 * Message handlers for VaakSeva WhatsApp client.
 *
 * Routes incoming WhatsApp messages to the appropriate FastAPI endpoint:
 *   - Text messages -> POST /api/query
 *   - Voice notes (OGG audio) -> POST /api/voice-query (multipart)
 *   - Other media -> inform user of supported types
 *
 * Response handling:
 *   - Text response: send as WhatsApp text message
 *   - Voice response: send TTS audio as WhatsApp voice note
 *   - Errors: send Hindi error message
 */

import { WASocket, proto } from '@whiskeysockets/baileys';
import pino from 'pino';
import * as fs from 'fs';
import * as path from 'path';
import { VaakSevaAPI } from './api';
import { downloadAudio, getPhoneNumber } from './utils';

const logger = pino({ level: 'info' });

// Hindi error messages for user-facing failures
const HINDI_ERRORS = {
  server_error: 'क्षमा करें, अभी सेवा उपलब्ध नहीं है। कृपया थोड़ी देर बाद प्रयास करें।',
  audio_error: 'क्षमा करें, आपका ऑडियो संदेश प्राप्त नहीं हो सका। कृपया दोबारा भेजें।',
  unsupported:
    'VaakSeva केवल हिंदी में टेक्स्ट और वॉयस संदेश स्वीकार करता है। सरकारी योजनाओं के बारे में प्रश्न पूछें।',
  greeting:
    'नमस्ते! मैं VaakSeva हूं। आप मुझसे हिंदी में सरकारी योजनाओं के बारे में पूछ सकते हैं। ' +
    'उदाहरण: "पीएम किसान योजना क्या है?" या "मेरे लिए कौन सी योजनाएं हैं?"',
};

export class MessageHandler {
  private sock: WASocket;
  private api: VaakSevaAPI;

  constructor(sock: WASocket) {
    this.sock = sock;
    this.api = new VaakSevaAPI();
  }

  async handleMessage(message: proto.IWebMessageInfo): Promise<void> {
    const jid = message.key.remoteJid!;
    const phoneNumber = getPhoneNumber(jid);
    const msgContent = message.message;

    if (!msgContent) return;

    logger.info({ jid, phoneNumber }, 'Received message');

    // Text message
    if (msgContent.conversation || msgContent.extendedTextMessage) {
      const text =
        msgContent.conversation || msgContent.extendedTextMessage?.text || '';

      // Handle greetings
      if (/^(hi|hello|नमस्ते|हेलो|start|help)\s*$/i.test(text.trim())) {
        await this.sendText(jid, HINDI_ERRORS.greeting);
        return;
      }

      await this.handleTextQuery(jid, phoneNumber, text);
      return;
    }

    // Voice note (OGG audio)
    if (msgContent.audioMessage) {
      const audio = msgContent.audioMessage;
      if (audio.ptt) {
        // PTT = Push-to-talk = voice note
        await this.handleVoiceNote(jid, phoneNumber, message);
      } else {
        await this.sendText(jid, HINDI_ERRORS.unsupported);
      }
      return;
    }

    // Unsupported message type
    await this.sendText(jid, HINDI_ERRORS.unsupported);
  }

  private async handleTextQuery(
    jid: string,
    phoneNumber: string,
    text: string
  ): Promise<void> {
    try {
      // Show typing indicator
      await this.sock.sendPresenceUpdate('composing', jid);

      const result = await this.api.query(text, phoneNumber);

      await this.sendText(jid, result.response_text);
    } catch (error) {
      logger.error({ error, jid }, 'Text query failed');
      await this.sendText(jid, HINDI_ERRORS.server_error);
    } finally {
      await this.sock.sendPresenceUpdate('available', jid);
    }
  }

  private async handleVoiceNote(
    jid: string,
    phoneNumber: string,
    message: proto.IWebMessageInfo
  ): Promise<void> {
    let audioPath: string | null = null;

    try {
      await this.sock.sendPresenceUpdate('recording', jid);

      // Download the voice note
      audioPath = await downloadAudio(this.sock, message);

      // Send to FastAPI voice endpoint
      const result = await this.api.voiceQuery(audioPath, phoneNumber);

      // Send text response
      await this.sendText(jid, result.response_text);

      // Send voice response if available
      if (result.audio_response_path && fs.existsSync(result.audio_response_path)) {
        await this.sendVoiceNote(jid, result.audio_response_path);
      }
    } catch (error) {
      logger.error({ error, jid }, 'Voice query failed');
      await this.sendText(jid, HINDI_ERRORS.audio_error);
    } finally {
      // Clean up downloaded audio
      if (audioPath && fs.existsSync(audioPath)) {
        fs.unlinkSync(audioPath);
      }
      await this.sock.sendPresenceUpdate('available', jid);
    }
  }

  private async sendText(jid: string, text: string): Promise<void> {
    await this.sock.sendMessage(jid, { text });
  }

  private async sendVoiceNote(jid: string, audioPath: string): Promise<void> {
    const audioBuffer = fs.readFileSync(audioPath);
    await this.sock.sendMessage(jid, {
      audio: audioBuffer,
      mimetype: 'audio/ogg; codecs=opus',
      ptt: true, // Makes it appear as a voice note, not an audio file
    });
  }
}

// typing indicator shown while FastAPI processes the request
