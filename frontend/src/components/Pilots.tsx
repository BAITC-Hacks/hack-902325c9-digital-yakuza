import type { Pilot } from '../types/api';
import { displayValue } from '../utils/labels';
import { DataTable, Notice, number, percent, SegmentDetails } from './Ui';

export function Pilots({ pilots, loading }: { pilots: Pilot[]; loading: boolean }) {
  if (!pilots.length) return <Notice>{loading ? 'Ожидаем первые результаты пробных запусков…' : 'Пробных запусков пока нет.'}</Notice>;
  return <DataTable label="Пробные запуски"
    headers={['№', 'Гипотеза', 'Сегмент', 'Тариф', 'Канал', 'Клиентов', { label: 'Изменение дохода', help: 'Наблюдаемое относительное изменение дохода на клиента в пробном запуске. Например, 5% означает рост, а −5% — снижение.' }, 'Стоимость, у.е.']}
    rows={pilots.map((pilot) => [
      pilot.sequence, <span className="description">{pilot.hypothesis.replace(/tariff_(\d+)/g, 'тариф $1').replace(/ARPU/g, 'дохода на клиента').replace(/\bsms\b/g, 'СМС').replace(/\bpush\b/g, 'уведомления').replace(/\bdigital_ads\b/g, 'интернет-рекламу').replace(/\bcall\b/g, 'звонок')}</span>,
      <SegmentDetails filters={pilot.segment} />, displayValue(pilot.target_tariff),
      <span className="tag">{displayValue(pilot.channel)}</span>, number(pilot.pilot_size), percent(pilot.observed_lift), number(pilot.cost, 2),
    ])} />;
}
