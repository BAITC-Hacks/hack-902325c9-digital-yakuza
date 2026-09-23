import { useEffect, useRef, useState } from 'react';
import { api, ApiError, errorMessage } from '../api/client';
import type { AgentRun, DashboardStatus, Pilot } from '../types/api';

export function useAgentRun() {
  const [run, setRun] = useState<AgentRun | null>(null);
  const [pilots, setPilots] = useState<Pilot[]>([]);
  const [activeId, setActiveId] = useState<string>();
  const [revision, setRevision] = useState(0);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState('');
  const [pilotError, setPilotError] = useState('');
  const [startError, setStartError] = useState('');
  const startLock = useRef(false);

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;
    let selectedId = activeId;
    setLoading(true);

    async function poll() {
      let repeat = false;
      try {
        const next = await api.result(selectedId, controller.signal);
        if (controller.signal.aborted) return;
        selectedId = next.run_id;
        setRun(next);
        setError('');
        repeat = next.status === 'running';
        try {
          const response = await api.pilots(next.run_id, controller.signal);
          if (controller.signal.aborted) return;
          setPilots(response.pilots);
          setPilotError('');
        } catch (reason) {
          if (!controller.signal.aborted) setPilotError(errorMessage(reason));
        }
      } catch (reason) {
        if (controller.signal.aborted) return;
        if (reason instanceof ApiError && reason.status === 404 && !selectedId) {
          setRun(null);
          setPilots([]);
          setError('');
          setPilotError('');
        } else {
          setError(errorMessage(reason));
          repeat = true;
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
          if (repeat) timer = window.setTimeout(() => void poll(), 2000);
        }
      }
    }
    void poll();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [activeId, revision]);

  async function start(seed: number) {
    if (startLock.current || run?.status === 'running') return;
    startLock.current = true;
    setStarting(true);
    setStartError('');
    try {
      const next = await api.start(seed);
      setRun(next);
      setPilots([]);
      setPilotError('');
      setError('');
      setActiveId(next.run_id);
    } catch (reason) {
      setStartError(reason instanceof ApiError && reason.status === 409
        ? 'Уже выполняется запуск. Загружаем его состояние.'
        : errorMessage(reason) + ' Проверяем последний запуск перед повторной попыткой.');
      setLoading(true);
      setActiveId(undefined);
      setRevision((value) => value + 1);
    } finally {
      setStarting(false);
      startLock.current = false;
    }
  }

  const status: DashboardStatus = starting ? 'running'
    : run?.status === 'failed' ? 'error'
    : run?.status ?? (error ? 'error' : 'idle');

  return {
    run, pilots, loading, starting, status, error, pilotError, startError, start,
    refresh: () => { setStartError(''); setRevision((value) => value + 1); },
  };
}
