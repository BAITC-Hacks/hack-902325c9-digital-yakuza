import type { AgentRun } from '../types/api';
import { explanationReport, estimateLabel } from '../utils/explanation';
import { Notice, percent } from './Ui';

export function RiskInfo({ value }: { value?: Record<string, unknown> | null }) {
  if (!value || !Object.keys(value).length) return null;
  const labels: Record<string, string> = {
    mu: 'Оценка эффекта (mu)', lcb: 'Нижняя граница (LCB)', downside: 'Риск снижения', exposure_arpu: 'Доход под риском',
  };
  return <details className="risk-details"><summary>Оценка риска</summary><dl>
    {Object.entries(value).filter(([, item]) => item !== null && item !== undefined).map(([key, item]) =>
      <div key={key}><dt>{labels[key] ?? key}</dt><dd>{(key === 'mu' || key === 'lcb') && typeof item === 'number'
        ? percent(item) : typeof item === 'object' ? JSON.stringify(item) : String(item)}</dd></div>)}
  </dl></details>;
}

export function Explanation({ run, pending }: { run: AgentRun | null; pending: boolean }) {
  const report = explanationReport(run);
  const rendered = report?.rendered;
  const source = run?.explanation?.explanation?.source;
  const sourceLabel = source === 'llm' ? 'ИИ (luna)' : source === 'template' ? 'Шаблон' : 'Источник не указан';
  const stopReason = run?.stop_reason ?? run?.metrics?.stop_reason;
  const estimateSource = run?.estimate_source ?? run?.metrics?.estimate_source;
  const risk = run?.risk_info ?? run?.metrics?.risk_info;
  const hasContent = rendered && (rendered.summary || rendered.campaigns?.length || rendered.warnings?.length || rendered.next_steps?.length);

  return <section id="explanation" aria-labelledby="explanation-title">
    <div className="section-heading"><h2 id="explanation-title">Почему такой план</h2><span className="tag">{sourceLabel}</span></div>
    {pending && <Notice>План готов. Ожидаем объяснение от сервера…</Notice>}
    {!hasContent && !pending && <Notice>Объяснение недоступно</Notice>}
    {hasContent && <div className="panel explanation-panel">
      {rendered.summary && <p>{rendered.summary}</p>}
      {!!rendered.campaigns?.length && <div className="campaign-explanations">{rendered.campaigns.map((item) =>
        <article key={item.id}><h3>{item.id}</h3><p>{item.explanation}</p></article>)}</div>}
      {!!rendered.warnings?.length && <div className="explanation-warnings"><h3>На что обратить внимание</h3><ul>
        {rendered.warnings.map((item, index) => <li key={index}>{typeof item === 'string' ? item : item.text}</li>)}
      </ul></div>}
      {!!rendered.next_steps?.length && <div><h3>Следующие шаги</h3><ol>
        {rendered.next_steps.map((item, index) => <li key={index}>{typeof item === 'string' ? item : item.text}</li>)}
      </ol></div>}
    </div>}
    {(typeof stopReason === 'string' || typeof estimateSource === 'string' || (risk !== null && typeof risk === 'object')) &&
      <div className="plan-context">
        {typeof stopReason === 'string' && stopReason && <p>Причина остановки: {stopReason}</p>}
        {typeof estimateSource === 'string' && estimateSource && <p>Источник оценки: {estimateLabel(estimateSource)}</p>}
        {risk !== null && typeof risk === 'object' && <RiskInfo value={risk as Record<string, unknown>} />}
      </div>}
  </section>;
}
