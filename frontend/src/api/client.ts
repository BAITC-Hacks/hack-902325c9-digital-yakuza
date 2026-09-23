import type { AgentRun, CaseSummary, PilotsResponse } from '../types/api';
import { serverMessage } from '../utils/labels';

const baseUrl = import.meta.env.VITE_API_URL?.trim().replace(/\/+$/, '');

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Не удалось выполнить запрос.';
}

async function request<T>(path: string, options: RequestInit = {}, csv = false): Promise<T> {
  if (!baseUrl) throw new Error('Задайте VITE_API_URL в frontend/.env и перезапустите Vite.');
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 30000);
  const abort = () => controller.abort();
  options.signal?.addEventListener('abort', abort, { once: true });
  if (options.signal?.aborted) controller.abort();
  try {
    const response = await fetch(baseUrl + '/api' + path, {
      ...options,
      signal: controller.signal,
      headers: { Accept: csv ? 'text/csv' : 'application/json', ...options.headers },
    });
    if (!response.ok) {
      const body: unknown = await response.json().catch(() => null);
      const detail = body && typeof body === 'object' && 'detail' in body ? body.detail : null;
      throw new ApiError(response.status, typeof detail === 'string' ? serverMessage(detail) : 'Ошибка запроса: HTTP ' + response.status);
    }
    if (csv) {
      if (!response.headers.get('content-type')?.includes('text/csv')) {
        throw new Error('Сервер вернул неожиданный формат вместо CSV.');
      }
      return await response.blob() as T;
    }
    return await response.json() as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (controller.signal.aborted) {
      throw new Error(options.signal?.aborted ? 'Запрос отменён.' : 'Сервер не ответил за 30 секунд. Обновите данные.');
    }
    if (error instanceof TypeError) throw new Error('Нет соединения с сервером. Убедитесь, что он запущен, и повторите попытку.');
    throw error;
  } finally {
    window.clearTimeout(timeout);
    options.signal?.removeEventListener('abort', abort);
  }
}

const query = (runId?: string) => runId ? '?run_id=' + encodeURIComponent(runId) : '';

export const api = {
  summary: (signal?: AbortSignal) => request<CaseSummary>('/case/summary', { signal }),
  start: (seed: number) => request<AgentRun>('/agent/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ seed }),
  }),
  result: (runId?: string, signal?: AbortSignal) =>
    request<AgentRun>('/agent/result' + query(runId), { signal }),
  pilots: (runId: string, signal?: AbortSignal) =>
    request<PilotsResponse>('/agent/pilots' + query(runId), { signal }),
  submission: (runId: string) => request<Blob>('/agent/submission' + query(runId), {}, true),
};
