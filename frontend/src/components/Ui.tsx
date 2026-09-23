import type { ReactNode } from 'react';
import type { DashboardStatus } from '../types/api';

export function number(value: unknown, digits = 0): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toLocaleString('ru-RU', { maximumFractionDigits: digits }) : '—';
}

export function percent(value: unknown): string {
  return typeof value === 'number' && Number.isFinite(value) ? number(value * 100, 2) + '%' : '—';
}

export function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return <div className="stat"><span>{label}</span><strong>{value}</strong>{note && <small>{note}</small>}</div>;
}

export function Notice({ children, error = false, retry }: { children: ReactNode; error?: boolean; retry?: () => void }) {
  return <div className={'notice' + (error ? ' notice-error' : '')} role={error ? 'alert' : 'status'}>
    <span>{children}</span>{retry && <button className="button secondary" onClick={retry}>Повторить</button>}
  </div>;
}

export function Status({ status }: { status: DashboardStatus }) {
  return <span className={'status status-' + status}><i />{status}</span>;
}

export function DataTable({ headers, rows, label }: { headers: string[]; rows: ReactNode[][]; label: string }) {
  return <div className="table-scroll" tabIndex={0} role="region" aria-label={label}>
    <table><caption className="sr-only">{label}</caption>
      <thead><tr>{headers.map((header) => <th key={header} scope="col">{header}</th>)}</tr></thead>
      <tbody>{rows.map((cells, index) => <tr key={index}>{cells.map((cell, column) => <td key={column}>{cell}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}

export function SegmentDetails({ filters }: { filters: Record<string, unknown> }) {
  const entries = Object.entries(filters).filter(([, value]) => value !== null && value !== undefined && value !== '');
  if (!entries.length) return <span className="muted">Вся аудитория</span>;
  const primary = entries.find(([key]) => key === 'filter_arpu_segment');
  return <details className="segment-details">
    <summary>{primary ? String(primary[1]) : 'Фильтры'} · {entries.length}</summary>
    <dl>{entries.map(([key, value]) => <div key={key}><dt>{key.replace('filter_', '')}</dt><dd>{String(value)}</dd></div>)}</dl>
  </details>;
}
