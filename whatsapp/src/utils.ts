/**
 * Audio download and utility functions for VaakSeva WhatsApp client.
 */

import { WASocket, proto, downloadMediaMessage } from '@whiskeysockets/baileys';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

/**
 * Download a WhatsApp voice note to a temporary file.
 * Returns the path to the downloaded OGG file.
 */
export async function downloadAudio(
  sock: WASocket,
  message: proto.IWebMessageInfo
): Promise<string> {
  const buffer = await downloadMediaMessage(
    message,
    'buffer',
    {},
    { logger: undefined as any, reuploadRequest: sock.updateMediaMessage }
  );

  const tmpPath = path.join(os.tmpdir(), `vaakseva_audio_${Date.now()}.ogg`);
  fs.writeFileSync(tmpPath, buffer as Buffer);
  return tmpPath;
}

/**
 * Extract the phone number from a WhatsApp JID.
 * JID format: "919876543210@s.whatsapp.net"
 * Returns: "+919876543210"
 */
export function getPhoneNumber(jid: string): string {
  const number = jid.split('@')[0];
  return `+${number}`;
}

/**
 * Check if a JID is a group (ends with @g.us)
 */
export function isGroup(jid: string): boolean {
  return jid.endsWith('@g.us');
}

/**
 * Format bytes to human readable string
 */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
