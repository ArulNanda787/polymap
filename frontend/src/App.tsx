import { useEffect, useMemo, useRef, useState } from "react";
import GlobeGraph, { type GlobeHandle } from "./GlobeGraph";
import {
  defaultFilters,
  edgePasses,
  KIND_COLOR,
  KIND_LABEL,
  type Filters,
  type Kind,
} from "./graphData";
import type { Graph } from "./lib/adapter";
import { fmtVolume } from "./lib/format";
import { useMarkets, useOpenMarket } from "./lib/useRunGraph";
import type { Market } from "./lib/types";
import HeroGlobe from "./HeroGlobe";

type Screen = "landing" | "markets" | "explorer";
type IconName =
  | "arrow"
  | "back"
  | "calendar"
  | "check"
  | "chevron"
  | "clock"
  | "filter"
  | "minus"
  | "pause"
  | "play"
  | "plus"
  | "reset"
  | "search"
  | "trend"
  | "wallet"
  | "x";

const Icon = ({ name, size = 18 }: { name: IconName; size?: number }) => {
  const paths: Record<IconName, React.ReactNode> = {
    arrow: <><path d="M5 12h14" /><path d="m13 6 6 6-6 6" /></>,
    back: <><path d="m15 18-6-6 6-6" /></>,
    calendar: <><rect x="3" y="5" width="18" height="16" rx="3" /><path d="M16 3v4M8 3v4M3 10h18" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    chevron: <path d="m9 18 6-6-6-6" />,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    filter: <><path d="M4 6h16M7 12h10M10 18h4" /></>,
    minus: <path d="M5 12h14" />,
    pause: <><path d="M9 5v14M15 5v14" /></>,
    play: <path d="m8 5 11 7-11 7V5Z" />,
    plus: <path d="M12 5v14M5 12h14" />,
    reset: <><path d="M20 11a8 8 0 1 0-2 5.3" /><path d="M20 5v6h-6" /></>,
    search: <><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></>,
    trend: <><path d="m4 16 5-5 4 4 7-8" /><path d="M15 7h5v5" /></>,
    wallet: <><path d="M4 7a3 3 0 0 1 3-3h11v16H6a2 2 0 0 1-2-2V7Z" /><path d="M4 8h14M15 13h3" /></>,
    x: <path d="m6 6 12 12M18 6 6 18" />,
  };
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {paths[name]}
    </svg>
  );
};

const Logo = ({ onClick }: { onClick?: () => void }) => (
  <button className="logo" onClick={onClick} aria-label="Polymap home">
    <span className="logo-mark">
      <i />
      <i />
      <i />
      <i />
    </span>
    <span>POLYMAP</span>
  </button>
);

const Button = ({
  children,
  onClick,
  kind = "primary",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  kind?: "primary" | "secondary" | "ghost";
}) => (
  <button className={`button button-${kind}`} onClick={onClick}>
    {children}
  </button>
);

const Landing = ({ onStart }: { onStart: () => void }) => (
  <main className="landing">
    <nav className="top-nav shell">
      <Logo />
      <div className="nav-links">
        <a href="#how">How it works</a>
        <a href="#signals">Signals</a>
        <Button onClick={onStart}>Open Polymap <Icon name="arrow" size={16} /></Button>
      </div>
    </nav>

    <section className="hero shell">
      <div className="eyebrow"><span /> Prediction-market intelligence</div>
      <h1>See the market.<br /><em>Map the motive.</em></h1>
      <p>Polymap analyzes wallet trading behavior to uncover recurring, persistent, and tightly synchronized relationships across prediction markets.</p>
      <div className="hero-actions">
        <Button onClick={onStart}>Explore the map <Icon name="arrow" size={17} /></Button>
        <span>No account required</span>
      </div>
      <div className="hero-visual" aria-label="Preview of wallet relationship graph">
        <div className="visual-head">
          <div><span className="status-dot" /> LIVE NETWORK</div>
          <span>286 wallets · 414 relationships</span>
        </div>
        <div className="hero-visual" aria-label="Preview of wallet relationship graph">
        <div className="visual-head">
          <div><span className="status-dot" /> LIVE NETWORK</div>
          <span>286 wallets · 414 relationships</span>
        </div>
        <HeroGlobe />
        <div className="preview-legend" aria-hidden="true">
          {(["both", "persistent", "tight", "temporal"] as const).map((k) => (
            <span key={k}><i style={{ borderColor: KIND_COLOR[k], background: `${KIND_COLOR[k]}33` }} />{KIND_LABEL[k]}</span>
          ))}
        </div>
        <div className="preview-card">
          <span className="wallet-avatar">0x</span>
          <div><strong>0x71F...92A</strong><small>Highly connected wallet</small></div>
          <b>94</b>
        </div>
      </div>
        <div className="preview-card">
          <span className="wallet-avatar">0x</span>
          <div><strong>0x71F...92A</strong><small>Highly connected wallet</small></div>
          <b>94</b>
        </div>
      </div>
    </section>

    <section className="proof-strip">
      <div className="shell proof-grid">
        <div><b>2.4M+</b><span>Trades analyzed</span></div>
        <div><b>48K</b><span>Wallets mapped</span></div>
        <div><b>16K</b><span>Relationships found</span></div>
        <div><b>84</b><span>Active markets</span></div>
      </div>
    </section>

    <section className="how shell" id="how">
      <div className="section-intro">
        <span className="section-kicker">BEYOND THE ORDER BOOK</span>
        <h2>Markets show outcomes.<br />We show <em>coordination.</em></h2>
      </div>
      <div className="feature-grid" id="signals">
        <article><span>01</span><div className="feature-icon"><Icon name="trend" /></div><h3>Temporal signals</h3><p>Find wallets that consistently trade the same markets within meaningful time windows.</p></article>
        <article><span>02</span><div className="feature-icon"><Icon name="wallet" /></div><h3>Persistent relationships</h3><p>Separate one-off coincidences from repeat behavior that persists across markets.</p></article>
        <article><span>03</span><div className="feature-icon"><Icon name="filter" /></div><h3>Investigate freely</h3><p>Filter, zoom, and trace individual interactions without losing the wider network.</p></article>
      </div>
    </section>
  </main>
);

const MarketSelection = ({ onBack, onSelect }: { onBack: () => void; onSelect: (m: Market) => void }) => {
  const [query, setQuery] = useState("");
  const { markets, error } = useMarkets();
  const visible = markets.filter((m) => (m.question ?? "").toLowerCase().includes(query.toLowerCase()));
  return (
    <main className="market-page">
      <nav className="app-nav shell">
        <Logo onClick={onBack} />
        <span className="step-label">MARKET DISCOVERY</span>
        <button className="icon-button" onClick={onBack}><Icon name="x" /></button>
      </nav>
      <section className="market-content shell">
        <button className="back-link" onClick={onBack}><Icon name="back" /> Back to overview</button>
        <div className="market-heading">
          <div><span className="section-kicker">STEP 01</span><h1>Choose a market<br />to <em>investigate.</em></h1></div>
          <p>Select an active prediction market to reveal its wallet network and relationship signals.</p>
        </div>
        <div className="market-tools">
          <label className="search-field"><Icon name="search" /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search markets..." /></label>
          <button className="filter-button"><Icon name="filter" /> All categories <Icon name="chevron" size={15} /></button>
        </div>
        {error && <p className="empty-note">Could not load markets: {error}</p>}
        <div className="market-grid">
          {visible.map((market, index) => (
            <button className="market-card" key={market.gamma_id} onClick={() => onSelect(market)}>
              <div className="market-card-top"><span className="market-symbol violet">PM</span><span className="market-category">{market.closed ? "Closed" : "Open"}</span><Icon name="arrow" /></div>
              <h2>{market.question ?? market.slug}</h2>
              <div className="market-stats">
                <span><small>VOLUME</small><b>{fmtVolume(market.volume)}</b></span>
                <span><small>ENDS</small><b>{market.end_date?.slice(0, 10) ?? "—"}</b></span>
              </div>
              <div className="activity-line"><i style={{ width: `${78 - Math.min(index, 10) * 6}%` }} /></div>
            </button>
          ))}
        </div>
      </section>
    </main>
  );
};

const timelineBars = (seed: number) =>
  Array.from({ length: 18 }, (_, i) => 24 + ((Math.sin(seed * 0.001 + i * 1.7) + 1) / 2) * 68);

const Explorer = ({ market, graph, onMarkets }: { market: string; graph: Graph; onMarkets: () => void }) => {
  const { nodes, edges, adjacency, defaultSelected } = graph;
  const [selected, setSelected] = useState(defaultSelected);
  const [selectedEdge, setSelectedEdge] = useState<number | null>(null);
  const [query, setQuery] = useState("");
  const [filters, setFilters] = useState<Filters>(defaultFilters);
  const [autoRotate, setAutoRotate] = useState(true);
  const [panelOpen, setPanelOpen] = useState(() => window.innerWidth > 760);
  const globe = useRef<GlobeHandle>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const selectedNode = nodes[selected];
  const shownEdges = useMemo(() => edges.filter((e) => edgePasses(e, filters)).length, [filters, edges]);
  const strongest = useMemo(
    () =>
      adjacency[selected]
        .map((ei) => ({ ei, other: edges[ei].a === selected ? edges[ei].b : edges[ei].a }))
        .sort((p, q) => edges[q.ei].interactions - edges[p.ei].interactions)
        .slice(0, 5),
    [selected, edges, adjacency],
  );

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        searchRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const chooseNode = (index: number, focus = false) => {
    setSelected(index);
    setSelectedEdge(null);
    setPanelOpen(true);
    if (focus) globe.current?.focus(index);
  };

  const chooseEdge = (edgeIndex: number) => {
    setSelectedEdge(edgeIndex);
    setSelected(edges[edgeIndex].a);
    setPanelOpen(true);
  };

  const search = (value: string) => {
    setQuery(value);
    if (value.length < 3) return;
    const match = nodes.findIndex((node) => node.id.toLowerCase().includes(value.toLowerCase()));
    if (match >= 0) chooseNode(match, true);
  };

  const resetFilters = () => setFilters(defaultFilters);
  const edge = selectedEdge !== null ? edges[selectedEdge] : null;
  const legend: [Kind, string][] = [["both", "Persistent + tight"], ["persistent", "Persistent"], ["tight", "Tight"], ["temporal", "Temporal link"]];

  return (
    <main className="explorer">
      <header className="explorer-nav">
        <Logo onClick={onMarkets} />
        <button className="market-switcher" onClick={onMarkets}><span className="market-symbol violet">PO</span><span><small>ACTIVE MARKET</small><b>{market}</b></span><Icon name="chevron" /></button>
        <div className="explorer-summary"><span><i className="live-dot" /> LIVE</span><b>{nodes.length}</b><small>WALLETS</small><b>{edges.length}</b><small>RELATIONSHIPS</small></div>
      </header>
      <div className="explorer-body">
        <aside className="filter-panel">
          <div className="panel-title"><span>NETWORK FILTERS</span><Icon name="filter" /></div>
          <label className="control-label">Relationship type</label>
          <div className="segmented">
            {([["all", "All"], ["persistent", "Persistent"], ["tight", "Tight"]] as const).map(([value, text]) => (
              <button key={value} className={filters.type === value ? "active" : ""} onClick={() => setFilters({ ...filters, type: value })}>{text}</button>
            ))}
          </div>
          <label className="control-label"><span>Min. interactions</span><b>{filters.minInteractions}+</b></label>
          <input className="range" type="range" min="1" max="8" value={filters.minInteractions} onChange={(event) => setFilters({ ...filters, minInteractions: Number(event.target.value) })} />
          <div className="range-labels"><span>1</span><span>8+</span></div>
          <label className="control-label"><span>TGNN score</span><b>{(filters.minScore / 100).toFixed(2)}+</b></label>
          <input className="range" type="range" min="0" max="95" value={filters.minScore} onChange={(event) => setFilters({ ...filters, minScore: Number(event.target.value) })} />
          <div className="range-labels"><span>0</span><span>1.0</span></div>
          <label className="control-label">Time range</label>
          <button className="select-control"><Icon name="calendar" /><span>Last 30 days</span><Icon name="chevron" /></button>
          <div className="divider" />
          <div className="legend">
            <span className="control-label">LEGEND</span>
            {legend.map(([kind, text]) => (<div key={kind}><i className="legend-dot" style={{ borderColor: KIND_COLOR[kind], background: `${KIND_COLOR[kind]}33` }} /> {text}</div>))}
            <div><i className="legend-dot" style={{ borderColor: "#3ddc97", background: "#3ddc9733" }} /> Linked to selection</div>
            <div><i className="legend-dot" style={{ borderColor: "#7e8fd0", background: "#2a3568" }} /> Below filter threshold</div>
          </div>
          <button className="reset-filters" onClick={resetFilters}><Icon name="reset" /> Reset filters</button>
        </aside>

        <section className="graph-area">
          <div className="graph-toolbar">
            <label className="graph-search"><Icon name="search" /><input ref={searchRef} placeholder="Find wallet address" value={query} onChange={(event) => search(event.target.value)} /><kbd>⌘ K</kbd></label>
            <div className="graph-actions">
              <button onClick={() => setAutoRotate(!autoRotate)} aria-label={autoRotate ? "Pause rotation" : "Resume rotation"} aria-pressed={autoRotate} className={autoRotate ? "on" : ""}><Icon name={autoRotate ? "pause" : "play"} /></button>
              <button onClick={() => globe.current?.zoomBy(1.25)} aria-label="Zoom in"><Icon name="plus" /></button>
              <button onClick={() => globe.current?.zoomBy(0.8)} aria-label="Zoom out"><Icon name="minus" /></button>
              <button onClick={() => globe.current?.reset()} aria-label="Recenter globe"><Icon name="reset" /></button>
            </div>
          </div>
          <GlobeGraph
            graph={graph}
            ref={globe}
            selected={selected}
            selectedEdge={selectedEdge}
            filters={filters}
            autoRotate={autoRotate}
            onSelectNode={(index) => chooseNode(index)}
            onSelectEdge={chooseEdge}
          />
          <div className="graph-meta"><span><i /> {nodes.length} wallets · {shownEdges} relationships above threshold</span><span>Source: screened pairs from the polymap API</span></div>
          <div className="graph-hint">Drag to rotate · Drag a wallet to pull it · Shift-drag to pan · Scroll to zoom · Double-click to recenter</div>
        </section>

        {panelOpen && (
          <aside className="detail-panel">
            <div className="detail-head">
              <div><span className="section-kicker">{edge ? "RELATIONSHIP ANALYSIS" : "WALLET DETAILS"}</span><h2>{edge ? "Trading relationship" : selectedNode.id}</h2></div>
              <button className="icon-button" onClick={() => setPanelOpen(false)} aria-label="Close panel"><Icon name="x" /></button>
            </div>
            {edge ? (
              <>
                <div className="wallet-pair">
                  <button onClick={() => chooseNode(edge.a, true)}><span className="wallet-avatar">0x</span><b>{nodes[edge.a].id}</b><small>source</small></button>
                  <span className="pair-link"><i /><i /><i /></span>
                  <button onClick={() => chooseNode(edge.b, true)}><span className="wallet-avatar dark">0x</span><b>{nodes[edge.b].id}</b><small>target</small></button>
                </div>
                <div className="signal-banner" style={{ background: `${KIND_COLOR[edge.kind]}22`, color: "#17201f" }}><Icon name="check" /><div><b>{KIND_LABEL[edge.kind]}</b><span>{edge.score >= 0.8 ? "High-confidence relationship" : edge.score >= 0.65 ? "Moderate-confidence relationship" : "Below confidence threshold"}</span></div><strong>{edge.score.toFixed(2)}</strong></div>
                <div className="stat-row"><div><small>INTERACTIONS</small><b>{edge.interactions}</b></div><div><small>AVG. TIME GAP</small><b>{edge.gap}</b></div><div><small>DISTINCT TIMES</small><b>{edge.shared}</b></div></div>
                <div className="timeline-head"><span>INTERACTION TIMELINE</span><span><i /> Wallet A <i /> Wallet B</span></div>
                <div className="timeline-chart">
                  <div className="chart-bars">
                    {timelineBars(edge.seed).map((height, index) => <i key={index} className={index % 3 === 0 ? "bar-dark" : ""} style={{ height: `${height}%` }} />)}
                  </div>
                  <div className="chart-dates"><span>May 01</span><span>May 15</span><span>May 30</span></div>
                </div>
                <div className="activity-list">
                  <div><span className="activity-icon buy">B</span><p><b>Both wallets bought YES</b><small>Price difference 0.3% · {edge.gap} apart</small></p><time>2h ago</time></div>
                  <div><span className="activity-icon sell">S</span><p><b>Mirrored position exit</b><small>${(edge.interactions * 1.1).toFixed(1)}k combined volume</small></p><time>1d ago</time></div>
                  <div><span className="activity-icon buy">B</span><p><b>Both wallets bought NO</b><small>Price difference 0.8% · {edge.gap} apart</small></p><time>3d ago</time></div>
                </div>
              </>
            ) : (
              <>
                <div className="wallet-profile"><span className="wallet-avatar dark">0x</span><div><b>{selectedNode.id}</b><span>Active since Jan 2025</span></div></div>
                <div className="stat-row"><div><small>TRADES</small><b>{selectedNode.trades}</b></div><div><small>VOLUME</small><b>{selectedNode.volume}</b></div><div><small>RELATIONSHIPS</small><b>{adjacency[selected].length}</b></div></div>
                <div className="signal-banner"><Icon name="trend" /><div><b>{selectedNode.score >= 70 ? "High network influence" : selectedNode.score >= 45 ? "Moderate network influence" : "Low network influence"}</b><span>{adjacency[selected].length === 0 ? "No linked wallets in this sample" : `Linked to ${adjacency[selected].length} wallet${adjacency[selected].length === 1 ? "" : "s"}`}</span></div><strong>{selectedNode.score}</strong></div>
                <div className="timeline-head"><span>STRONGEST CONNECTIONS</span></div>
                <div className="connection-list">
                  {strongest.length === 0 && <p className="empty-note">This wallet trades independently. No relationships detected.</p>}
                  {strongest.map(({ ei, other }) => (
                    <button key={ei} onClick={() => chooseNode(other, true)} onDoubleClick={() => chooseEdge(ei)}>
                      <span className="wallet-avatar">0x</span>
                      <p><b>{nodes[other].id}</b><small>{edges[ei].interactions} interactions · {KIND_LABEL[edges[ei].kind]}</small></p>
                      <strong>{edges[ei].score.toFixed(2)}</strong>
                    </button>
                  ))}
                </div>
              </>
            )}
          </aside>
        )}
      </div>
    </main>
  );
};

const Loading = ({ logs, error, onBack }: { logs: string[]; error?: string; onBack: () => void }) => (
  <main className="market-page">
    <section className="market-content shell">
      <button className="back-link" onClick={onBack}><Icon name="back" /> Back to markets</button>
      <h1>{error ? "Something went wrong" : "Analyzing market…"}</h1>
      {error && <p className="empty-note">{error}</p>}
      <pre style={{ maxHeight: 360, overflow: "auto", fontSize: 12, opacity: 0.7 }}>{logs.slice(-40).join("\n")}</pre>
    </section>
  </main>
);

export default function App() {
  const [screen, setScreen] = useState<Screen>("landing");
  const [market, setMarket] = useState<Market | null>(null);
  const open = useOpenMarket(market);
  const toMarkets = () => { setMarket(null); setScreen("markets"); };

  if (screen === "landing") return <Landing onStart={() => setScreen("markets")} />;
  if (screen === "markets" || !market)
    return <MarketSelection onBack={() => setScreen("landing")} onSelect={(m) => { setMarket(m); setScreen("explorer"); }} />;
  if (open.phase === "ready")
    return <Explorer key={open.run} market={market.question ?? open.run} graph={open.graph} onMarkets={toMarkets} />;
  return <Loading logs={"logs" in open ? open.logs : []} error={open.phase === "error" ? open.message : undefined} onBack={toMarkets} />;
}
