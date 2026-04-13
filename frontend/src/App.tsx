import { useEffect, useState, useRef } from "react";
import { Activity, CircleDollarSign, TrendingUp, Users, Settings2, PackageSearch } from "lucide-react";
import { cn } from "./lib/utils";

type Prediction = {
  purchase_probability: number;
  bayes_probability: number;
  predicted_revenue: number;
  final_prediction: string;
};

type VisitorPayload = {
  visitor_id: string;
  current_page: string;
  add_to_cart_count: number;
  features: any;
  prediction: Prediction;
};

export default function App() {
  const [visitors, setVisitors] = useState<VisitorPayload[]>([]);
  const [metrics, setMetrics] = useState({ total_visitors: 0, revenue: 0, conv_rate: 0 });
  const [selectedVisitor, setSelectedVisitor] = useState<VisitorPayload | null>(null);

  useEffect(() => {
    const ws = new WebSocket("ws://127.0.0.1:8000/ws/live_visitors");
    
    ws.onmessage = (event) => {
      try {
        const data: VisitorPayload = JSON.parse(event.data);
        
        setVisitors((prev) => {
          const filtered = prev.filter((v) => v.visitor_id !== data.visitor_id);
          const next = [data, ...filtered].slice(0, 15);
          return next;
        });

        setMetrics((prev) => {
          let revDelta = data.prediction.predicted_revenue;
          if (data.prediction.final_prediction === "Window Shopper" && data.prediction.predicted_revenue < 10) {
            revDelta = 0; // Don't count very low spend window shoppers
          }
          return {
            total_visitors: prev.total_visitors + 1,
            revenue: prev.revenue + revDelta,
            conv_rate: (prev.conv_rate * 9 + data.prediction.purchase_probability) / 10
          };
        });

      } catch (err) {
        console.error(err);
      }
    };

    return () => ws.close();
  }, []);

  return (
    <div className="min-h-screen bg-[#050505] text-gray-200 flex flex-col font-sans relative overflow-hidden">
      {/* Header */}
      <header className="glass-header px-6 py-4 flex items-center justify-between sticky top-0 z-40">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-yellow-500 to-orange-400 flex items-center justify-center shadow-[0_0_15px_rgba(245,158,11,0.4)]">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <h1 className="font-display font-bold text-lg tracking-wider text-white">MERCHANT COMMAND CENTER</h1>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-xs font-mono tracking-widest text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-500 live-pulse"></span>
            LIVE INTELLIGENCE
          </div>
        </div>
      </header>

      <main className="flex-1 p-6 flex flex-col lg:flex-row gap-6">
        
        {/* Left Column - Main Feed */}
        <div className="flex-1 flex flex-col gap-6">
          {/* Top Metrics Row */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <MetricCard 
              title="Active Sessions" 
              value={visitors.length.toString()} 
              icon={<Users className="w-4 h-4 text-blue-400" />} 
            />
            <MetricCard 
              title="Pipeline Revenue" 
              value={`$${metrics.revenue.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`} 
              valueClass="text-yellow-400"
              icon={<CircleDollarSign className="w-4 h-4 text-yellow-400" />} 
            />
            <MetricCard 
              title="Avg Conv. Prob" 
              value={`${(metrics.conv_rate * 100).toFixed(1)}%`} 
              icon={<TrendingUp className="w-4 h-4 text-emerald-400" />} 
            />
          </div>

          {/* Feed Container */}
          <div className="glass flex-1 flex flex-col overflow-hidden min-h-[400px]">
            <div className="px-5 py-4 border-b border-white/10 flex items-center justify-between">
              <h2 className="font-mono text-xs tracking-widest text-gray-400 uppercase">Visitor Intelligence Feed</h2>
              <span className="text-xs text-gray-500">{visitors.length} tracking</span>
            </div>
            
            <div className="overflow-y-auto flex-1 p-2 space-y-1">
              {visitors.length === 0 && (
                <div className="h-full flex flex-col items-center justify-center text-gray-500">
                  <PackageSearch className="w-8 h-8 mb-2 opacity-50" />
                  <p className="font-mono text-sm">Awaiting visitor signals...</p>
                </div>
              )}
              {visitors.map((v) => (
                <VisitorCard 
                  key={v.visitor_id} 
                  visitor={v} 
                  onClick={() => setSelectedVisitor(v)}
                  active={selectedVisitor?.visitor_id === v.visitor_id}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Right Column - Side Panel */}
        <div className="w-full lg:w-[360px] h-full">
          {selectedVisitor ? (
            <VisitorDetails visitor={selectedVisitor} onClose={() => setSelectedVisitor(null)} />
          ) : (
            <div className="glass h-full p-6 flex flex-col items-center justify-center text-center border-dashed border-white/20">
              <Settings2 className="w-10 h-10 text-gray-600 mb-4" />
              <h3 className="text-gray-400 font-medium mb-1">No Visitor Selected</h3>
              <p className="text-xs text-gray-500">Select a visitor from the live feed to view deep predictive insights & session tracing.</p>
            </div>
          )}
        </div>

      </main>
    </div>
  );
}

// Sub-components

function MetricCard({ title, value, icon, valueClass = "text-white" }: { title: string, value: string, icon: React.ReactNode, valueClass?: string }) {
  return (
    <div className="metric-card flex flex-col gap-2">
      <div className="flex items-center justify-between text-gray-400">
        <span className="text-xs font-mono uppercase tracking-widest">{title}</span>
        {icon}
      </div>
      <div className={cn("text-3xl font-display font-bold", valueClass)}>{value}</div>
    </div>
  );
}

function VisitorCard({ visitor, onClick, active }: { visitor: VisitorPayload, onClick: () => void, active: boolean }) {
  const p = visitor.prediction;
  
  let tagColor = "text-gray-400 bg-gray-500/10 border-gray-500/20";
  let barColor = "bg-gray-600";
  
  if (p.final_prediction === "High Intent") {
    tagColor = "text-yellow-400 bg-yellow-400/10 border-yellow-400/20";
    barColor = "bg-yellow-400";
  } else if (p.final_prediction === "At Risk") {
    tagColor = "text-emerald-400 bg-emerald-400/10 border-emerald-400/20";
    barColor = "bg-emerald-400";
  }

  const probPct = Math.round(p.purchase_probability * 100);

  return (
    <div 
      onClick={onClick}
      className={cn(
        "animate-slide-in flex items-center p-3 rounded-lg cursor-pointer transition-all duration-300 group",
        active ? "bg-white/10 shadow-lg border border-white/20" : "hover:bg-white/5 border border-transparent"
      )}
    >
      <div className="w-[80px] text-xs font-mono text-gray-400">{visitor.visitor_id}</div>
      
      <div className="flex-1 flex items-center gap-3 px-4">
        <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
          <div className={cn("h-full rounded-full transition-all duration-1000", barColor)} style={{ width: `${probPct}%` }}></div>
        </div>
        <div className="text-xs font-mono text-gray-300 w-8">{probPct}%</div>
      </div>

      <div className="w-[120px] flex justify-center">
        <span className={cn("text-[10px] uppercase font-bold tracking-wider py-1 px-2 rounded border", tagColor)}>
          {p.final_prediction}
        </span>
      </div>

      <div className="w-[80px] text-right text-sm font-display font-bold text-gray-200">
        ${p.predicted_revenue.toFixed(0)}
      </div>
    </div>
  );
}

function VisitorDetails({ visitor, onClose }: { visitor: VisitorPayload, onClose: () => void }) {
  const p = visitor.prediction;
  const f = visitor.features;

  return (
    <div className="glass h-full flex flex-col overflow-hidden animate-slide-in">
      <div className="p-4 border-b border-white/10 flex justify-between items-center bg-white/5">
        <div>
          <h2 className="font-mono text-sm tracking-widest text-white">{visitor.visitor_id}</h2>
          <p className="text-xs text-gray-400 mt-1">Active Session Details</p>
        </div>
        <button onClick={onClose} className="text-gray-400 hover:text-white pb-2 text-xl">&times;</button>
      </div>

      <div className="p-5 flex-1 overflow-y-auto space-y-6">
        
        {/* Core Prediction */}
        <div>
          <h3 className="text-[10px] uppercase tracking-widest text-gray-500 mb-3">AI Prediction</h3>
          <div className="bg-[#0a0a0b] border border-white/10 rounded p-4 space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Class</span>
              <span className="text-sm font-bold text-yellow-400">{p.final_prediction}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Expected Value</span>
              <span className="text-xl font-display font-bold text-white">${p.predicted_revenue.toFixed(2)}</span>
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-gray-400">Decision Tree</span>
                <span>{(p.purchase_probability * 100).toFixed(1)}%</span>
              </div>
              <div className="h-1 bg-white/10 rounded-full overflow-hidden">
                <div className="h-full bg-yellow-400 rounded-full" style={{ width: `${p.purchase_probability * 100}%` }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-gray-400">Bayesian Probability</span>
                <span>{(p.bayes_probability * 100).toFixed(1)}%</span>
              </div>
              <div className="h-1 bg-white/10 rounded-full overflow-hidden">
                <div className="h-full bg-blue-400 rounded-full" style={{ width: `${p.bayes_probability * 100}%` }}></div>
              </div>
            </div>
          </div>
        </div>

        {/* Live Tracking */}
        <div>
          <h3 className="text-[10px] uppercase tracking-widest text-gray-500 mb-3">Live Feed</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between border-b border-white/5 py-1">
              <span className="text-gray-400">Current URL</span>
              <span className="font-mono text-xs">{visitor.current_page}</span>
            </div>
            <div className="flex justify-between border-b border-white/5 py-1">
              <span className="text-gray-400">Add to Cart</span>
              <span className="font-mono text-emerald-400">{visitor.add_to_cart_count} events</span>
            </div>
          </div>
        </div>

        {/* Session Stats */}
        <div>
          <h3 className="text-[10px] uppercase tracking-widest text-gray-500 mb-3">Session Feature State</h3>
          <div className="grid grid-cols-2 gap-2">
            <StatBox label="Prod. Views" value={f.ProductRelated.toPrecision(3)} />
            <StatBox label="Admin Views" value={f.Administrative.toPrecision(3)} />
            <StatBox label="Bounce Rate" value={f.BounceRates.toFixed(3)} />
            <StatBox label="Page Value" value={`${f.PageValues.toFixed(1)}`} />
          </div>
        </div>

      </div>
    </div>
  );
}

function StatBox({ label, value }: { label: string, value: string }) {
  return (
    <div className="bg-[#0a0a0b] border border-white/5 rounded p-2 text-center">
      <div className="font-mono text-[10px] text-gray-500 mb-1 tracking-widest uppercase">{label}</div>
      <div className="text-sm font-semibold">{value}</div>
    </div>
  );
}
