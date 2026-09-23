import type { ReactNode } from 'react';
import type { DashboardStatus } from '../types/api';
import { displayValue, segmentLabels } from '../utils/labels';
import { Help } from './Help';

export function number(value: unknown, digits = 0): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toLocaleString('ru-RU', { maximumFractionDigits: digits }) : '—';
}

export function percent(value: unknown): string {
  return typeof value === 'number' && Number.isFinite(value) ? number(value * 100, 2) + '%' : '—';
}

export function Stat({ label, value, note, help }: { label: string; value: string; note?: string; help?: string }) {
  return <div className="stat"><span className="stat-label">{label}{help && <Help label={label}>{help}</Help>}</span><strong>{value}</strong>{note && <small>{note}</small>}</div>;
}

export function Notice({ children, error = false, retry }: { children: ReactNode; error?: boolean; retry?: () => void }) {
  return <div className={'notice' + (error ? ' notice-error' : '')} role={error ? 'alert' : 'status'}>
    <span>{children}</span>{retry && <button className="button secondary" onClick={retry}>Повторить</button>}
  </div>;
}

export function Status({ status }: { status: DashboardStatus }) {
  const labels = { idle: 'Готов к запуску', running: 'Выполняется', completed: 'Завершён', error: 'Ошибка' };
  return <span className={'status status-' + status}><i />{labels[status]}</span>;
}

export function DataTable({ headers, rows, label }: { headers: (string | { label: string; help: string })[]; rows: ReactNode[][]; label: string }) {
  return <div className="table-scroll" tabIndex={0} role="region" aria-label={label}>
    <table><caption className="sr-only">{label}</caption>
      <thead><tr>{headers.map((header) => <th key={typeof header === 'string' ? header : header.label} scope="col">
        {typeof header === 'string' ? header : <span className="stat-label">{header.label}<Help label={header.label}>{header.help}</Help></span>}
      </th>)}</tr></thead>
      <tbody>{rows.map((cells, index) => <tr key={index}>{cells.map((cell, column) => <td key={column}>{cell}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}

export function SegmentDetails({ filters }: { filters: Record<string, unknown> }) {
  const entries = Object.entries(filters).filter(([, value]) => value !== null && value !== undefined && value !== '');
  if (!entries.length) return <span className="muted">Вся аудитория</span>;
  const primary = entries.find(([key]) => key === 'filter_arpu_segment');
  return <details className="segment-details">
    <summary>{primary ? displayValue(primary[1]) + ' доход' : 'Условия отбора'}</summary>
    <dl>{entries.map(([key, value]) => <div key={key}><dt>{segmentLabels[key.replace('filter_', '')] ?? 'Условие отбора'}</dt><dd>{displayValue(value)}</dd></div>)}</dl>
  </details>;
}
