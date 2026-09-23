import { useEffect, useState } from 'react';
import { api, errorMessage } from '../api/client';
import { Campaigns } from '../components/Campaigns';
import { CaseOverview } from '../components/CaseOverview';
import { Pilots } from '../components/Pilots';
import { Notice, Status } from '../components/Ui';
import { useAgentRun } from '../hooks/useAgentRun';
import { useCaseSummary } from '../hooks/useCaseSummary';

export function Dashboard() {
  const summary = useCaseSummary();
  const agent = useAgentRun();
  const [seed, setSeed] = useState('42');
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState('');
  const [downloadedRun, setDownloadedRun] = useState('');
  const [downloadUrl, setDownloadUrl] = useState('');
  const numericSeed = Number(seed);
  const validSeed = seed.trim() !== '' && Number.isInteger(numericSeed) && numericSeed >= 0 && numericSeed <= 2147483647;
  const busy = agent.starting || agent.run?.status === 'running';

  useEffect(() => () => { if (downloadUrl) URL.revokeObjectURL(downloadUrl); }, [downloadUrl]);

  async function download() {
    if (!agent.run || downloading) return;
    const id = agent.run.run_id;
    setDownloading(true);
    setDownloadError('');
    try {
      const blob = await api.submission(id);
      if (!blob.size) throw new Error('Backend вернул пустой CSV.');
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'submission.csv';
      document.body.appendChild(link);
      link.click();
      link.remove();
      setDownloadUrl(url);
      setDownloadedRun(id);
    } catch (reason) {
      setDownloadError(errorMessage(reason));
    } finally {
      setDownloading(false);
    }
  }

  return <div className="layout">
    <aside className="sidebar">
      <a href="#overview" className="brand"><span className="brand-mark">H</span><span>HackAlem<small>DIGITAL YAKUZA</small></span></a>
      <span className="sidebar-label">WORKSPACE</span>
      <nav aria-label="Разделы dashboard">
        <a href="#overview"><span>01</span> Case Summary</a>
        <a href="#run"><span>02</span> Agent Run</a>
        <a href="#results"><span>03</span> Agent Result</a>
        <a href="#pilots"><span>04</span> Pilots</a>
        <a href="#submission"><span>05</span> Submission</a>
      </nav>
      <div className="sidebar-foot"><span className="live-dot" /> LOCAL WORKSPACE<small>Beeline · Campaign analytics</small></div>
    </aside>
    <main>
      <header className="topbar"><span>HackAlem / <strong>Analytics</strong></span><span className="environment">OFFICIAL MOCK</span></header>
      <div className="content">
        <div className="page-heading"><div><span className="eyebrow">BEELINE CASE</span><h1>Campaign dashboard</h1>
          <p>От аудитории и пилотов — к финальной стратегии.</p></div>
          <a className="button secondary" href="#run">К запуску агента ↗</a>
        </div>
        <Notice>Синтетические данные · официальная mock-среда. Эффект кампаний — оценка модели, не результат судейства.</Notice>

        <section id="overview" aria-labelledby="overview-title">
          <div className="section-heading"><div><span className="section-number">01</span><h2 id="overview-title">Case Summary</h2></div>
            <button className="text-button" disabled={summary.loading} onClick={summary.reload}>Обновить данные</button></div>
          {summary.loading && <Notice>Загружаем аудиторию, тарифы и лимиты…</Notice>}
          {summary.error && <Notice error retry={summary.reload}>{summary.error}</Notice>}
          {summary.data && <CaseOverview data={summary.data} />}
        </section>

        <section id="run" aria-labelledby="run-title">
          <div className="section-heading"><div><span className="section-number">02</span><h2 id="run-title">Agent Run</h2></div><Status status={agent.status} /></div>
          <div className="panel run-panel">
            <div><h3>{busy ? 'Агент выполняет стратегию' : 'Запустить анализ кампаний'}</h3>
              <p className="muted">Агент проверяет гипотезы на пилотах и собирает финальные кампании.</p>
              <p className="run-id">{agent.run ? 'Run ID: ' + agent.run.run_id : 'Запуск ещё не создан'}</p>
              {agent.run && <p className="muted">Seed: {agent.run.seed} · Начало: {new Date(agent.run.started_at).toLocaleString('ru-RU')}</p>}
            </div>
            <form onSubmit={(event) => { event.preventDefault(); if (validSeed) { setDownloadError(''); void agent.start(numericSeed); } }}>
              <label htmlFor="seed">Seed</label>
              <div className="run-controls"><input id="seed" type="number" min="0" max="2147483647" step="1" required value={seed}
                disabled={busy} onChange={(event) => setSeed(event.target.value)} />
                <button className="button primary" disabled={busy || agent.loading || !!agent.error || !validSeed} type="submit">
                  {busy ? 'Выполняется…' : 'Запустить агента'}</button></div>
              {!validSeed && <small className="error-text">Введите целое число от 0 до 2147483647.</small>}
            </form>
          </div>
          {agent.loading && <Notice>Проверяем состояние запуска…</Notice>}
          {busy && <Notice>Статус и пилоты обновляются каждые 2 секунды. Можно оставаться на этой странице.</Notice>}
          {agent.startError && <Notice error>{agent.startError}</Notice>}
          {agent.error && <Notice error retry={agent.refresh}>{agent.error}</Notice>}
          {agent.run?.status === 'failed' && <Notice error>{agent.run.error ?? 'Агент завершился с ошибкой. Попробуйте новый запуск.'}</Notice>}
        </section>

        <section id="results" aria-labelledby="results-title">
          <div className="section-heading"><div><span className="section-number">03</span><h2 id="results-title">Agent Result</h2></div>
            <button className="text-button" disabled={agent.loading || agent.starting} onClick={agent.refresh}>Обновить результат</button></div>
          <Campaigns run={agent.run} />
        </section>

        <section id="pilots" aria-labelledby="pilots-title">
          <div className="section-heading"><div><span className="section-number">04</span><h2 id="pilots-title">Pilots</h2><span className="count">{agent.pilots.length}</span></div></div>
          {agent.pilotError && <Notice error retry={agent.refresh}>{agent.pilotError}</Notice>}
          <Pilots pilots={agent.pilots} loading={busy || agent.loading} />
        </section>

        <section id="submission" aria-labelledby="submission-title">
          <div className="section-heading"><div><span className="section-number">05</span><h2 id="submission-title">Submission</h2></div></div>
          <div className="panel submission-panel"><div><h3>submission.csv</h3>
            <p className="muted">Финальные кампании выбранного запуска в формате backend.</p>
            <small>{agent.run?.status === 'completed' ? 'Готов к скачиванию · Run ' + agent.run.run_id : 'Файл доступен после успешного завершения агента.'}</small></div>
            <button className="button primary" onClick={() => void download()} disabled={agent.run?.status !== 'completed' || busy || downloading}>
              {downloading ? 'Скачиваем…' : 'Скачать submission.csv ↓'}</button></div>
          {downloadError && <Notice error>{downloadError}</Notice>}
          {downloadedRun && downloadedRun === agent.run?.run_id && <Notice>
            CSV получен. Если скачивание не началось: <a className="text-button" href={downloadUrl} download="submission.csv">Сохранить CSV</a>
          </Notice>}
        </section>
        <footer>HackAlem · Digital Yakuza <span>Локальная аналитика / React + FastAPI</span></footer>
      </div>
    </main>
  </div>;
}
