import { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

export function Help({ label, children }: { label: string; children: string }) {
  const id = useId();
  const button = useRef<HTMLButtonElement>(null);
  const timer = useRef<number | undefined>(undefined);
  const [position, setPosition] = useState<{ left: number; top: number; above: boolean } | null>(null);

  function show() {
    window.clearTimeout(timer.current);
    const rect = button.current?.getBoundingClientRect();
    if (!rect) return;
    const width = Math.min(280, window.innerWidth - 24);
    const above = rect.bottom + 160 > window.innerHeight;
    setPosition({
      left: Math.max(12, Math.min(rect.left, window.innerWidth - width - 12)),
      top: above ? rect.top - 6 : rect.bottom + 6,
      above,
    });
  }

  function hide() {
    timer.current = window.setTimeout(() => setPosition(null), 120);
  }

  useEffect(() => {
    const close = () => setPosition(null);
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') close(); };
    window.addEventListener('scroll', close, true);
    window.addEventListener('resize', close);
    window.addEventListener('keydown', escape);
    return () => {
      window.clearTimeout(timer.current);
      window.removeEventListener('scroll', close, true);
      window.removeEventListener('resize', close);
      window.removeEventListener('keydown', escape);
    };
  }, []);

  return <>
    <button ref={button} type="button" className="help-button" aria-label={'О показателе: ' + label}
      aria-describedby={position ? id : undefined} onMouseEnter={show} onMouseLeave={hide}
      onFocus={show} onBlur={hide} onClick={show}>!</button>
    {position && createPortal(<span id={id} role="tooltip" className="help-tooltip"
      style={{ left: position.left, top: position.top, transform: position.above ? 'translateY(-100%)' : undefined }}
      onMouseEnter={() => window.clearTimeout(timer.current)} onMouseLeave={hide}>{children}</span>, document.body)}
  </>;
}
