import { useEffect, useState } from 'react';
import { api, errorMessage } from '../api/client';
import type { CaseSummary } from '../types/api';

export function useCaseSummary() {
  const [data, setData] = useState<CaseSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError('');
    api.summary(controller.signal)
      .then((result) => { if (!controller.signal.aborted) setData(result); })
      .catch((reason: unknown) => { if (!controller.signal.aborted) setError(errorMessage(reason)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [revision]);

  return { data, loading, error, reload: () => setRevision((value) => value + 1) };
}
