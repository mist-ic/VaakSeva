/**
 * HTTP client for VaakSeva FastAPI backend.
 *
 * Handles all communication between the Baileys WhatsApp client
 * and the Python FastAPI pipeline.
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import FormData from 'form-data';
import * as fs from 'fs';
import pino from 'pino';

const logger = pino({ level: 'info' });

const BACKEND_URL = process.env.WHATSAPP_BACKEND_URL || 'http://localhost:8080';
const TIMEOUT_MS = 120_000; // 2 minutes (LLM can be slow)
const MAX_RETRIES = 2;

export interface QueryResult {
  request_id: string;
  response_text: string;
  timings: Record<string, number | null>;
}

export interface VoiceQueryResult {
  request_id: string;
  transcript: string;
  transcript_confidence: number;
  response_text: string;
  audio_response_path: string | null;
}

export class VaakSevaAPI {
  private client: AxiosInstance;

  constructor(baseUrl?: string) {
    this.client = axios.create({
      baseURL: baseUrl || BACKEND_URL,
      timeout: TIMEOUT_MS,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  async query(message: string, phoneNumber: string): Promise<QueryResult> {
    return this._withRetry(async () => {
      const response = await this.client.post<QueryResult>('/api/query', {
        message,
        phone_number: phoneNumber,
      });
      return response.data;
    });
  }

  async voiceQuery(audioPath: string, phoneNumber: string): Promise<VoiceQueryResult> {
    return this._withRetry(async () => {
      const form = new FormData();
      form.append('audio', fs.createReadStream(audioPath), {
        filename: 'voice.ogg',
        contentType: 'audio/ogg',
      });

      const response = await this.client.post<VoiceQueryResult>(
        `/api/voice-query?phone_number=${encodeURIComponent(phoneNumber)}`,
        form,
        { headers: form.getHeaders() }
      );
      return response.data;
    });
  }

  private async _withRetry<T>(fn: () => Promise<T>): Promise<T> {
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= MAX_RETRIES + 1; attempt++) {
      try {
        return await fn();
      } catch (error) {
        const axiosError = error as AxiosError;
        lastError = error as Error;

        if (axiosError.response && axiosError.response.status < 500) {
          // Client error — don't retry
          throw error;
        }

        if (attempt <= MAX_RETRIES) {
          const delayMs = 1000 * attempt;
          logger.warn({ attempt, delayMs }, 'API call failed, retrying...');
          await sleep(delayMs);
        }
      }
    }

    throw lastError;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
