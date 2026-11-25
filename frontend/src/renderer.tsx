import { Component, useMemo, useState } from "react";
import type { ReactNode, ChangeEvent } from "react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import type { A2UIMessage } from "./types";

const chartPalette = ["#6c8df5", "#35b69a", "#e6a04b", "#e36d79", "#9271d8", "#49a4ca"];


type Comp = Record<string, any>;
class SurfaceErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(error: Error) { console.error("Inline GenUI render error", error); }
  render() { return this.state.failed ? <div className="render-error">This generated UI could not render. The text answer is still available above.</div> : this.props.children; }
}
export function ChatSurface({ messages, onAsk }: { messages: A2UIMessage[]; onAsk: (prompt: string) => void }) {
  const { components } = useMemo(() => {
    const byId: Record<string, Comp> = {};
    let surfaceId = "";
    for (const message of messages || []) {
      if (message.createSurface) { surfaceId = message.createSurface.surfaceId; }
      if (message.updateComponents?.surfaceId === surfaceId) {
        for (const component of message.updateComponents.components || []) {
          if (component?.id) byId[component.id] = component;
        }
      }
    }
    return { components: byId };
  }, [messages]);
  if (!Object.keys(components).length) return null;
  const root = (Object.values(components) as Comp[]).find((c) => c.id === "chat_genui_root");
  return <div className="genui-surface" aria-label="Generated inline interface">
    {root ? <SurfaceErrorBoundary><RenderComponent component={root} components={components} onAsk={onAsk} /></SurfaceErrorBoundary> : <div className="genui-empty">UI surface is missing its root component.</div>}
  </div>;
}


function RenderComponent({ component, components, onAsk, depth = 0 }: { component: Comp; components: Record<string, Comp>; onAsk: (prompt: string) => void; depth?: number }) {
  if (depth > 8) return <div className="render-error">Component nesting limit reached.</div>;
  const renderChild = (id: string) => {
    const child = components[id];
    return child ? <RenderComponent key={id} component={child} components={components} onAsk={onAsk} depth={depth + 1}/> : <div key={id} className="render-error">Missing component: {id}</div>;
  };
  const children = (component.children || []).map((id: string) => renderChild(id));
  switch (component.component) {
    case "Column": return <div className="ui-column">{children}</div>;
    case "Row": return <div className="ui-row">{children}</div>;
    case "Card": return <section className="ui-card">{component.title && <div className="ui-card-title">{component.title}</div>}{component.child ? renderChild(component.child) : children}</section>;
    case "Text": return <UIText text={String(component.text ?? "")} variant={component.variant || "body"}/>;
    case "Metric": return <Metric component={component}/>;
    case "Chart": return <Chart component={component}/>;
    case "DataTable": return <DataTable component={component}/>;
    case "List": return <ul className="ui-list">{(component.items || []).map((item: string, i: number) => <li key={i}>{item}</li>)}</ul>;
    case "Slider": return <Slider component={component}/>;
    case "Calculator": return <Calculator component={component}/>;
    case "AskAgentButton": return <button className="ui-action" onClick={() => onAsk(String(component.prompt || ""))}>{component.label || "Ask agent"}<span>↗</span></button>;
    case "Callout": return <div className={`ui-callout ${component.tone || "info"}`}><strong>{component.title}</strong><p>{component.text}</p></div>;
    default: return <div className="render-error">Unsupported component in this catalog: {component.component}</div>;
  }
}


function UIText({ text, variant }: { text: string; variant: string }) {
  if (variant === "code") return <pre className="ui-code"><code>{text}</code></pre>;
  if (variant === "heading" || variant === "title") return <div className={`ui-text ${variant}`}>{text}</div>;
  if (variant === "caption") return <div className="ui-text caption">{text}</div>;
  return <div className="ui-text body">{text}</div>;
}

function Metric({ component }: { component: Comp }) {
  return <div className="ui-metric"><div className="metric-label">{component.label}</div><div className="metric-value">{component.value}</div>{component.detail && <div className="metric-detail">{component.detail}</div>}</div>;
}


function Chart({ component }: { component: Comp }) {
  const { chart_type, data, x_key, series } = component;
  const common = { data, margin: { top: 10, right: 12, bottom: 6, left: 0 } };
  const xAxis = <XAxis dataKey={x_key} tick={{ fontSize: 11 }} tickLine={false} axisLine={false}/>;
  const yAxis = <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={42}/>;
  const grid = <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)"/>;
  const tooltip = <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 10, color: "var(--text)" }} labelStyle={{ color: "var(--muted)" }}/>;
  const plot = chart_type === "bar" ? <BarChart {...common}>{grid}{xAxis}{yAxis}{tooltip}{series.map((s: Comp, i: number) => <Bar key={s.key} dataKey={s.key} name={s.label} fill={chartPalette[i % chartPalette.length]} radius={[4, 4, 0, 0]} maxBarSize={44}/>)}</BarChart>
    : chart_type === "area" ? <AreaChart {...common}>{grid}{xAxis}{yAxis}{tooltip}{series.map((s: Comp, i: number) => <Area key={s.key} dataKey={s.key} name={s.label} stroke={chartPalette[i % chartPalette.length]} fill={chartPalette[i % chartPalette.length]} fillOpacity={0.12} strokeWidth={2} type="monotone"/> )}</AreaChart>
    : chart_type === "scatter" ? <ScatterChart {...common}>{grid}{xAxis}{yAxis}{tooltip}{series.map((s: Comp, i: number) => <Scatter key={s.key} data={data.map((p: Comp) => ({ ...p, [x_key]: Number(p[x_key]) }))} dataKey={s.key} name={s.label} fill={chartPalette[i % chartPalette.length]}/>)}</ScatterChart>
    : <LineChart {...common}>{grid}{xAxis}{yAxis}{tooltip}{series.map((s: Comp, i: number) => <Line key={s.key} dataKey={s.key} name={s.label} stroke={chartPalette[i % chartPalette.length]} strokeWidth={2.4} type="monotone" dot={false} activeDot={{ r: 4 }}/>)}</LineChart>;
  return <div className="ui-chart-card"><div className="chart-title">{component.title || "Chart"}</div>{data?.length ? <div className="chart-wrap"><ResponsiveContainer width="100%" height={260}>{plot}</ResponsiveContainer></div> : <div className="chart-empty">No chart observations available.</div>}<div className="chart-legend">{series.map((s: Comp, i: number) => <span key={s.key}><i style={{ background: chartPalette[i % chartPalette.length] }}/>{s.label}</span>)}</div></div>;
}


function DataTable({ component }: { component: Comp }) {
  const columns = component.columns || [];
  const rows = component.rows || [];
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);
  const visibleRows = useMemo(() => {
    if (!sort) return rows;
    return [...rows].sort((a: Comp, b: Comp) => String(a[sort.key] ?? "").localeCompare(String(b[sort.key] ?? ""), undefined, { numeric: true }) * sort.dir);
  }, [rows, sort]);
  return <div className="ui-table-card"><div className="table-title">{component.title || "Data"}</div><div className="table-scroll"><table><thead><tr>{columns.map((col: Comp) => <th key={col.key}><button onClick={() => setSort((old) => old && old.key === col.key ? { key: col.key, dir: old.dir === 1 ? -1 : 1 } : { key: col.key, dir: 1 })}>{col.label}{sort && sort.key === col.key ? (sort.dir === 1 ? " ↑" : " ↓") : " ↕"}</button></th>)}</tr></thead><tbody>{visibleRows.map((row: Comp, i: number) => <tr key={i}>{columns.map((col: Comp) => <td key={col.key}>{row[col.key] ?? "—"}</td>)}</tr>)}</tbody></table></div><div className="table-count">{rows.length} row{rows.length === 1 ? "" : "s"}</div></div>;
}


function Slider({ component }: { component: Comp }) {
  const [value, setValue] = useState(Number(component.value));
  return <div className="ui-slider"><div className="slider-head"><span>{component.label}</span><strong>{value.toLocaleString(undefined, { maximumFractionDigits: 3 })}{component.unit ? ` ${component.unit}` : ""}</strong></div><input type="range" min={component.min} max={component.max} step={component.step} value={value} onChange={(e: ChangeEvent<HTMLInputElement>) => setValue(Number(e.target.value))}/><div className="slider-range"><span>{component.min}{component.unit ? ` ${component.unit}` : ""}</span><span>{component.max}{component.unit ? ` ${component.unit}` : ""}</span></div></div>;
}


function Calculator({ component }: { component: Comp }) {
  const [values, setValues] = useState<Record<string, number>>(() =>
    Object.fromEntries(component.fields.map((f: Comp) => [f.name, Number(f.value)]))
  );
  const result = useMemo(() => {
    const numbers = component.fields.map((f: Comp) => Number(values[f.name]));
    switch (component.operation) {
      case "energy_cost": return (values.power_kw * values.duration_hours * values.price_eur_mwh) / 1000;
      case "multiply": return numbers.reduce((a: number, b: number) => a * b, 1);
      case "add": return numbers.reduce((a: number, b: number) => a + b, 0);
      case "subtract": return numbers.slice(1).reduce((a: number, b: number) => a - b, numbers[0]);
      case "divide": return numbers.slice(1).reduce((a: number, b: number) => b === 0 ? NaN : a / b, numbers[0]);
      case "percentage": return numbers[1] === 0 ? NaN : numbers[0] / numbers[1] * 100;
      default: return NaN;
    }
  }, [component.fields, component.operation, values]);
  const resultText = Number.isFinite(result) ? result.toLocaleString(undefined, { maximumFractionDigits: 6 }) : "Cannot calculate";
  const reset = () => setValues(Object.fromEntries(component.fields.map((f: Comp) => [f.name, Number(f.value)])));
  return <div className="ui-calculator">
    <div className="calc-title">{component.title || "Interactive calculator"}</div>
    <div className="calc-fields">
      {component.fields.map((field: Comp) => <label key={field.name} className="calc-field">
        <span>{field.label}</span>
        <div className="calc-input-wrap">
          <input type="number" min={field.min} max={field.max} step={field.step} value={values[field.name]}
            onChange={(e: ChangeEvent<HTMLInputElement>) => {
              const next = Number(e.target.value);
              if (Number.isFinite(next)) setValues((prev) => ({
                ...prev,
                [field.name]: Math.min(Number(field.max), Math.max(Number(field.min), next)),
              }));
            }} />
          <small>{field.unit}</small>
        </div>
      </label>)}
    </div>
    <div className="calc-result">
      <div className="calc-result-label">{component.output_label}</div>
      <div className="calc-result-value">{resultText}{component.output_unit ? ` ${component.output_unit}` : ""}</div>
    </div>
    <button className="calc-reset" onClick={reset}>Reset values</button>
  </div>;
}
