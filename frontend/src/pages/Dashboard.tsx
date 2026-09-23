import { useEffect, useState } from 'react';
import { api, errorMessage } from '../api/client';
import { Campaigns } from '../components/Campaigns';
import { Explanation } from '../components/Explanation';
import { CaseOverview } from '../components/CaseOverview';
import { Pilots } from '../components/Pilots';
import { Notice, Status } from '../components/Ui';
import { Help } from '../components/Help';
import { serverMessage } from '../utils/labels';
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
      if (!blob.size) throw new Error('Сервер вернул пустой файл.');
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

      <nav aria-label="Разделы аналитики">
        <a href="#overview"><span>01</span> Обзор аудитории</a>
        <a href="#run"><span>02</span> Запуск анализа</a>
        <a href="#results"><span>03</span> Кампании</a>
        <a href="#explanation">Почему такой план</a>
        <a href="#pilots"><span>04</span> Пробные запуски</a>
        <a href="#submission"><span>05</span> Выгрузка</a>
      </nav>

    </aside>
    <main>
      <header className="topbar"><span>HackAlem / <strong>Аналитика</strong></span></header>
      <div className="content">
        <div className="page-heading"><div><h1>Аналитика кампаний</h1>
          <p>Аудитория, пробные запуски и результаты кампаний.</p></div>
          <a className="button secondary" href="#run">К запуску агента ↓</a>
        </div>
        <p className="simulation-note">Результаты моделирования <Help label="Результаты моделирования">Эффект кампаний оценивается в тестовой среде. Это не фактический доход и не итоговая оценка жюри.</Help></p>

        <section id="overview" aria-labelledby="overview-title">
          <div className="section-heading"><div><span className="section-number">01</span><h2 id="overview-title">Обзор аудитории</h2></div>
            <button className="text-button" disabled={summary.loading} onClick={summary.reload}>Обновить данные</button></div>
          {summary.loading && <Notice>Загружаем аудиторию, тарифы и лимиты…</Notice>}
          {summary.error && <Notice error retry={summary.reload}>{summary.error}</Notice>}
          {summary.data && <CaseOverview data={summary.data} />}
        </section>

        <section id="run" aria-labelledby="run-title">
          <div className="section-heading"><div><span className="section-number">02</span><h2 id="run-title">Запуск анализа</h2></div><Status status={agent.status} /></div>
          <div className="panel run-panel">
            <div><h3>{busy ? 'Агент выполняет стратегию' : 'Запустить анализ кампаний'}</h3>
              <p className="muted">Агент проверяет гипотезы на небольшой аудитории и собирает финальные кампании.</p>

              {agent.run && <p className="muted">Начало: {new Date(agent.run.started_at).toLocaleString('ru-RU')}</p>}
            </div>
            <form onSubmit={(event) => { event.preventDefault(); if (validSeed) { setDownloadError(''); void agent.start(numericSeed); } }}>
              <details className="run-settings"><summary>Параметры запуска</summary><div className="seed-setting"><label htmlFor="seed">Число случайности</label><Help label="Число случайности">При одинаковом числе и неизменных данных можно повторить условия моделирования. Для обычного запуска оставьте 42.</Help>
              <input id="seed" type="number" min="0" max="2147483647" step="1" required value={seed}
                disabled={busy} onChange={(event) => setSeed(event.target.value)} /></div></details><div className="run-controls">
                <button className="button primary" disabled={busy || agent.loading || !!agent.error || !validSeed} type="submit">
                  {busy ? 'Выполняется…' : 'Запустить агента'}</button></div>
              {!validSeed && <small className="error-text">Введите целое число от 0 до 2147483647.</small>}
            </form>
          </div>
          {agent.loading && <Notice>Проверяем состояние запуска…</Notice>}
          {busy && <Notice>Результаты обновляются автоматически.</Notice>}
          {agent.startError && <Notice error>{agent.startError}</Notice>}
          {agent.error && <Notice error retry={agent.refresh}>{agent.error}</Notice>}
          {agent.run?.status === 'failed' && <Notice error>{agent.run.error ? serverMessage(agent.run.error) : 'Агент завершился с ошибкой. Попробуйте новый запуск.'}</Notice>}
        </section>

        <section id="results" aria-labelledby="results-title">
          <div className="section-heading"><div><span className="section-number">03</span><h2 id="results-title">Кампании</h2></div>
            <button className="text-button" disabled={agent.loading || agent.starting} onClick={agent.refresh}>Обновить результат</button></div>
          <Campaigns run={agent.run} />
        </section>

        <Explanation run={agent.run} pending={agent.explanationPending} />

        <section id="pilots" aria-labelledby="pilots-title">
          <div className="section-heading"><div><span className="section-number">04</span><h2 id="pilots-title">Пробные запуски</h2><Help label="Пробные запуски">Агент проверяет предложение на небольшой группе клиентов, прежде чем включить его в итоговую стратегию.</Help><span className="count">{agent.pilots.length}</span></div></div>
          {agent.pilotError && <Notice error retry={agent.refresh}>{agent.pilotError}</Notice>}
          <Pilots pilots={agent.pilots} loading={busy || agent.loading} links={agent.run?.explanation?.campaign_pilot_links} />
        </section>

        <section id="submission" aria-labelledby="submission-title">
          <div className="section-heading"><div><span className="section-number">05</span><h2 id="submission-title">Выгрузка</h2></div></div>
          <div className="panel submission-panel"><div><h3>Файл с кампаниями</h3>
            <p className="muted">Финальные кампании в формате CSV для отправки решения.</p>
            <small>{agent.run?.status === 'completed' ? 'Готов к скачиванию' : 'Файл доступен после успешного завершения агента.'}</small></div>
            <button className="button primary" onClick={() => void download()} disabled={agent.run?.status !== 'completed' || busy || downloading}>
              {downloading ? 'Скачиваем…' : 'Скачать CSV ↓'}</button></div>
          {downloadError && <Notice error>{downloadError}</Notice>}
          {downloadedRun && downloadedRun === agent.run?.run_id && <Notice>
            CSV получен. Если скачивание не началось: <a className="text-button" href={downloadUrl} download="submission.csv">Сохранить CSV</a>
          </Notice>}
        </section>

      </div>
    </main>
  </div>;
}
