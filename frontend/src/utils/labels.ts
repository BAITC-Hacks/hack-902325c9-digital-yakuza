const values: Record<string, string> = {
  HIGH: 'Высокий', MID: 'Средний', MEDIUM: 'Средний', LOW: 'Низкий',
  HEAVY: 'Активное использование', LITE: 'Небольшое использование',
  NON_USER: 'Не пользуются', ACTIVE: 'Активные', INACTIVE: 'Неактивные',
  sms: 'СМС', push: 'Уведомления', digital_ads: 'Интернет-реклама', call: 'Звонок',
};

export const segmentLabels: Record<string, string> = {
  arpu_segment: 'Доход на клиента', data_segment: 'Интернет',
  call_segment: 'Звонки', current_tariff: 'Текущий тариф',
};

export function displayValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return 'Не указан';
  return String(value).split(';').map((part) => {
    const key = part.trim();
    return values[key] ?? key.replace(/^tariff_(\d+)$/, 'Тариф $1');
  }).join(', ');
}

export function serverMessage(message: string): string {
  const messages: Record<string, string> = {
    'Agent run not found': 'Запуск не найден.',
    'An agent run is already in progress': 'Другой запуск уже выполняется.',
    'Submission is not ready': 'Файл ещё не готов. Дождитесь завершения анализа.',
    'Beeline data package unavailable': 'Данные кейса временно недоступны.',
    'Agent exceeded the 600 second limit': 'Превышено время выполнения: 10 минут.',
    'Backend stopped during agent execution': 'Сервер остановлен во время анализа. Запустите анализ повторно.',
    'Agent execution failed; see backend logs': 'Анализ завершился с ошибкой. Подробности доступны в журнале сервера.',
  };
  return messages[message] ?? (/[а-яё]/i.test(message) ? message : 'Не удалось выполнить операцию. Повторите попытку.');
}

export function campaignLabel(name: string): string {
  const match = /^(LOW|MID|HIGH)_tariff_\d+_(\d+)$/.exec(name);
  return match ? 'Кампания ' + match[2] + ' · ' + displayValue(match[1]).toLocaleLowerCase('ru-RU') + ' доход' : name;
}
