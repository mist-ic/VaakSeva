/**
 * VaakSeva WhatsApp Client
 *
 * Uses Baileys (whiskeysockets/baileys) to connect to WhatsApp Web via WebSocket.
 * Maintains persistent auth state so QR code scan is only needed once.
 *
 * Architecture:
 *   WhatsApp Web <-> Baileys (Node.js) <-> HTTP <-> FastAPI (Python)
 *
 * The Baileys process is intentionally separate from the Python backend:
 *   - Different tech stack requirements (Node.js WebSocket client)
 *   - Separation of concerns: messaging layer vs AI pipeline
 *   - Allows independent restarting of either component
 *
 * IMPORTANT: Baileys is for development/demo. For production at scale,
 * use WhatsApp Business Cloud API (official Meta API) with webhook architecture.
 */

import makeWASocket, {
  DisconnectReason,
  fetchLatestBaileysVersion,
  makeInMemoryStore,
  useMultiFileAuthState,
} from '@whiskeysockets/baileys';
import { Boom } from '@hapi/boom';
import pino from 'pino';
import * as fs from 'fs';
import * as path from 'path';
import { MessageHandler } from './handlers';

const AUTH_DIR = path.join(__dirname, '..', 'auth');
const logger = pino({ level: 'info' });

async function connectToWhatsApp(): Promise<void> {
  // Ensure auth directory exists
  fs.mkdirSync(AUTH_DIR, { recursive: true });

  const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);
  const { version } = await fetchLatestBaileysVersion();

  logger.info({ version }, 'Using Baileys version');

  const store = makeInMemoryStore({ logger: pino({ level: 'silent' }) });

  const sock = makeWASocket({
    version,
    logger: pino({ level: 'silent' }) as any,
    printQRInTerminal: true,
    auth: state,
    browser: ['VaakSeva', 'Chrome', '118.0'],
    markOnlineOnConnect: false,
    generateHighQualityLinkPreview: false,
    syncFullHistory: false,
  });

  store.bind(sock.ev);

  const handler = new MessageHandler(sock);

  // Handle credential updates (save after QR scan)
  sock.ev.on('creds.update', saveCreds);

  // Handle connection state changes
  sock.ev.on('connection.update', async (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      logger.info('Scan the QR code above to connect to WhatsApp');
    }

    if (connection === 'close') {
      const shouldReconnect =
        (lastDisconnect?.error as Boom)?.output?.statusCode !== DisconnectReason.loggedOut;

      logger.warn(
        { reason: (lastDisconnect?.error as Boom)?.output?.statusCode },
        'Connection closed'
      );

      if (shouldReconnect) {
        logger.info('Reconnecting in 5 seconds...');
        setTimeout(connectToWhatsApp, 5000);
      } else {
        logger.info('Logged out. Delete auth/ directory and restart to re-scan QR code.');
      }
    } else if (connection === 'open') {
      logger.info('WhatsApp connection established successfully');
    }
  });

  // Handle incoming messages
  sock.ev.on('messages.upsert', async (event) => {
    if (event.type !== 'notify') return;

    for (const message of event.messages) {
      // Skip messages from self
      if (message.key.fromMe) continue;
      // Skip status updates
      if (message.key.remoteJid === 'status@broadcast') continue;

      try {
        await handler.handleMessage(message);
      } catch (error) {
        logger.error({ error, messageKey: message.key }, 'Error handling message');
      }
    }
  });
}

// Start
connectToWhatsApp().catch((err) => {
  logger.error({ err }, 'Fatal error starting WhatsApp client');
  process.exit(1);
});
