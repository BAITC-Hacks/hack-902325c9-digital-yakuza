import type { Pilot } from '../types/api';
import { DataTable, Notice, number, percent, SegmentDetails } from './Ui';

export function Pilots({ pilots, loading }: { pilots: Pilot[]; loading: boolean }) {
  if (!pilots.length) return <Notice>{loading ? 'Ожидаем первые результаты пилотов…' : 'Пилотных запусков пока нет.'}</Notice>;
  return <DataTable label="Пилотные запуски"
    headers={['№', 'Гипотеза', 'Сегмент', 'Тариф', 'Канал', 'Размер', 'Observed lift', 'Стоимость, у.е.']}
    rows={pilots.map((pilot) => [
      pilot.sequence, <span className="description">{pilot.hypothesis}</span>,
      <SegmentDetails filters={pilot.segment} />, pilot.target_tariff,
      <span className="tag">{pilot.channel}</span>, number(pilot.pilot_size), percent(pilot.observed_lift), number(pilot.cost, 2),
    ])} />;
}
