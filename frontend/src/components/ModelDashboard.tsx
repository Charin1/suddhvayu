import React, { useState, useEffect } from 'react';
import {
    LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Area, AreaChart, Legend
} from 'recharts';
import { Activity, BarChart2, TrendingUp, RefreshCw, AlertCircle, CheckCircle2, FlaskConical } from 'lucide-react';
import DataPipelineStatus from './DataPipelineStatus';

// --- Types ---

interface ForecastResponse {
    forecast: { date: string; value: number }[];
    metadata: { mae: number; last_trained: string };
    recent_performance: { date: string; actual: number; predicted: number }[];
}

const POLLUTANTS = ['AQI', 'PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3'];
const MODELS = ['XGBoost', 'Linear Regression', 'SVR', 'Random Forest', 'ANN', 'LSTM'];

const ModelDashboard: React.FC = () => {
    const [activeTab, setActiveTab] = useState<'trends' | 'distribution' | 'training'>('training');
    const [city] = useState('Ahmedabad');

    // --- Trends State ---
    const [historyData, setHistoryData] = useState<any[]>([]);
    const [selectedPollutants, setSelectedPollutants] = useState<string[]>(['AQI', 'PM2.5']);
    const [trendsLoading, setTrendsLoading] = useState(true);

    // --- Distribution State ---
    const [distData, setDistData] = useState<any[]>([]);
    const [distPollutant, setDistPollutant] = useState('AQI');
    const [distLoading, setDistLoading] = useState(false);

    // --- Forecast State ---
    const [forecastTarget, setForecastTarget] = useState('AQI');
    const [selectedModel, setSelectedModel] = useState('XGBoost');
    const [forecastData, setForecastData] = useState<ForecastResponse | null>(null);
    const [training, setTraining] = useState(false);
    const [predicting, setPredicting] = useState(false);
    const [trainStatus, setTrainStatus] = useState<{ type: 'success' | 'error', msg: string } | null>(null);

    // --- Fetch History (Trends) ---
    useEffect(() => {
        const fetchHistory = async () => {
            setTrendsLoading(true);
            try {
                const res = await fetch(`/api/v1/history/${city}`);
                if (res.ok) {
                    const json = await res.json();
                    setHistoryData(json.data);
                }
            } catch (e) {
                console.error("Fetch history failed", e);
            } finally {
                setTrendsLoading(false);
            }
        };
        fetchHistory(); // Always fetch for context
    }, [city]);

    // --- Fetch Distribution ---
    useEffect(() => {
        const fetchDist = async () => {
            setDistLoading(true);
            try {
                const res = await fetch(`/api/v1/distribution/${city}?pollutant=${distPollutant}`);
                if (res.ok) {
                    const json = await res.json();
                    const values = json.values;
                    const bins = 20;
                    const min = Math.min(...values);
                    const max = Math.max(...values);
                    const width = (max - min) / bins;
                    const histogram = Array.from({ length: bins }, (_, i) => {
                        const low = min + i * width;
                        const high = low + width;
                        const count = values.filter((v: number) => v >= low && (i === bins - 1 ? v <= high : v < high)).length;
                        return { range: `${Math.round(low)}-${Math.round(high)}`, count, low };
                    });
                    setDistData(histogram);
                }
            } catch (e) {
                console.error("Fetch dist failed", e);
            } finally {
                setDistLoading(false);
            }
        };
        if (activeTab === 'distribution') fetchDist();
    }, [city, distPollutant, activeTab]);

    // --- Forecast Actions ---
    const handleTrain = async () => {
        setTraining(true);
        setTrainStatus(null);
        try {
            const res = await fetch('/api/v1/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_type: selectedModel, city, target: forecastTarget })
            });
            if (res.ok) {
                setTrainStatus({ type: 'success', msg: `Training started for ${selectedModel}...` });
                setTimeout(() => handlePredict(), 3000);
            } else {
                setTrainStatus({ type: 'error', msg: 'Training failed to start.' });
            }
        } catch (e) {
            setTrainStatus({ type: 'error', msg: 'Network error during training.' });
        } finally {
            setTraining(false);
        }
    };

    const handlePredict = async () => {
        setPredicting(true);
        try {
            const res = await fetch('/api/v1/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_type: selectedModel, city, days: 7, target: forecastTarget })
            });
            if (res.ok) {
                const json = await res.json();
                setForecastData(json);
                setTrainStatus(null);
            } else {
                const err = await res.json();
                setTrainStatus({ type: 'error', msg: err.detail || 'Prediction failed' });
            }
        } catch (e) {
            setTrainStatus({ type: 'error', msg: 'Network error during prediction.' });
        } finally {
            setPredicting(false);
        }
    };

    // Auto-load forecast if tab is active and not loaded
    useEffect(() => {
        if (activeTab === 'training' && !forecastData && !predicting && !trainStatus) {
            handlePredict();
        }
    }, [activeTab]);


    return (
        <div className="grid grid-cols-12 gap-6 pb-10">
            {/* LEFT COLUMN: Controls & Status */}
            <div className="col-span-12 lg:col-span-4 space-y-6">
                <DataPipelineStatus />

                {/* Training Controls */}
                <div className="rounded-[20px] bg-dark-card border border-border p-6 shadow-sm">
                    <div className="flex items-center gap-3 mb-6">
                        <div className="p-2 bg-primary/10 rounded-lg text-primary border border-primary/20">
                            <FlaskConical size={20} />
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-text-primary">Model Experiments</h3>
                            <p className="text-xs text-text-muted">Train & Validate Models</p>
                        </div>
                    </div>

                    <div className="space-y-4">
                        <div className="space-y-2">
                            <label className="text-xs font-semibold text-text-muted uppercase">Target Pollutant</label>
                            <select
                                value={forecastTarget}
                                onChange={(e) => { setForecastTarget(e.target.value); setForecastData(null); }}
                                className="w-full bg-dark-base border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-primary"
                            >
                                {POLLUTANTS.map(p => <option key={p} value={p}>{p}</option>)}
                            </select>
                        </div>
                        <div className="space-y-2">
                            <label className="text-xs font-semibold text-text-muted uppercase">Algorithm</label>
                            <select
                                value={selectedModel}
                                onChange={(e) => { setSelectedModel(e.target.value); setForecastData(null); }}
                                className="w-full bg-dark-base border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-primary"
                            >
                                {MODELS.map(m => <option key={m} value={m}>{m}</option>)}
                            </select>
                        </div>

                        <div className="pt-2">
                            <button
                                onClick={handleTrain}
                                disabled={training}
                                className="w-full flex items-center justify-center gap-2 bg-primary hover:bg-primary/90 text-dark-base font-bold px-4 py-3 rounded-xl text-sm transition-colors disabled:opacity-50"
                            >
                                {training ? <RefreshCw className="animate-spin w-4 h-4" /> : <RefreshCw className="w-4 h-4" />}
                                {training ? 'Training Network...' : 'Start Training Run'}
                            </button>
                        </div>

                        {/* Status Message */}
                        {trainStatus && (
                            <div className={`p-3 rounded-lg text-sm flex items-center gap-2 ${trainStatus.type === 'error' ? 'bg-aqi-danger/10 text-aqi-danger' : 'bg-aqi-good/10 text-aqi-good'}`}>
                                {trainStatus.type === 'error' ? <AlertCircle size={16} /> : <CheckCircle2 size={16} />}
                                {trainStatus.msg}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* RIGHT COLUMN: Visualizations */}
            <div className="col-span-12 lg:col-span-8 space-y-6">
                {/* Navigation Tabs (Internal) */}
                <div className="flex bg-dark-card p-1 rounded-xl border border-border w-fit">
                    {[
                        { id: 'training', label: 'Evaluation & Forecast', icon: TrendingUp },
                        { id: 'trends', label: 'Feature Trends', icon: Activity },
                        { id: 'distribution', label: 'Data Distribution', icon: BarChart2 },
                    ].map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id as any)}
                            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === tab.id
                                    ? 'bg-primary/10 text-primary shadow-sm'
                                    : 'text-text-muted hover:text-text-primary'
                                }`}
                        >
                            <tab.icon size={16} />
                            {tab.label}
                        </button>
                    ))}
                </div>

                {/* TAB CONTENT: TRAINING & EVAL */}
                {activeTab === 'training' && (
                    <div className="rounded-[20px] bg-dark-card border border-border p-6 shadow-sm min-h-[500px]">
                        {forecastData ? (
                            <>
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                                    <div className="p-4 bg-dark-base rounded-xl border border-border">
                                        <div className="text-text-muted text-xs mb-1">Model MAE</div>
                                        <div className="text-xl font-bold text-primary">{forecastData.metadata.mae ? forecastData.metadata.mae.toFixed(2) : 'N/A'}</div>
                                    </div>
                                    <div className="p-4 bg-dark-base rounded-xl border border-border">
                                        <div className="text-text-muted text-xs mb-1">Last Trained</div>
                                        <div className="text-sm font-medium text-text-primary">{forecastData.metadata.last_trained || 'Never'}</div>
                                    </div>
                                    <div className="p-4 bg-dark-base rounded-xl border border-border">
                                        <div className="text-text-muted text-xs mb-1">Validation Set</div>
                                        <div className="text-xl font-bold text-secondary">14 Days</div>
                                    </div>
                                </div>

                                <div className="h-[300px] w-full border border-border rounded-xl p-4 bg-dark-base/50 mb-6">
                                    <h4 className="text-sm font-semibold text-text-primary mb-4">Recursive Forecast Debug (T+1 to T+7)</h4>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart data={forecastData.forecast}>
                                            <defs>
                                                <linearGradient id="colorValueModel" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="#2BD42E" stopOpacity={0.3} />
                                                    <stop offset="95%" stopColor="#2BD42E" stopOpacity={0} />
                                                </linearGradient>
                                            </defs>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
                                            <XAxis dataKey="date" tick={{ fill: '#8B949E', fontSize: 10 }} tickFormatter={(val) => new Date(val).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric' })} axisLine={false} tickLine={false} />
                                            <YAxis tick={{ fill: '#8B949E', fontSize: 12 }} axisLine={false} tickLine={false} />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#161B22', borderColor: '#30363d', color: '#E6EDF3', borderRadius: '8px' }}
                                            />
                                            <Area type="monotone" dataKey="value" stroke="#2BD42E" fill="url(#colorValueModel)" />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>

                                {/* Recent Performance Table */}
                                {forecastData.recent_performance && forecastData.recent_performance.length > 0 && (
                                    <div className="overflow-hidden rounded-xl border border-border">
                                        <div className="bg-dark-base p-3 border-b border-border text-sm font-semibold text-text-primary">
                                            Validation Set Performance (Actual vs Predicted)
                                        </div>
                                        <div className="max-h-[300px] overflow-y-auto">
                                            <table className="w-full text-sm text-left">
                                                <thead className="text-xs text-text-muted uppercase bg-dark-base sticky top-0">
                                                    <tr>
                                                        <th className="px-4 py-2">Date</th>
                                                        <th className="px-4 py-2">Actual</th>
                                                        <th className="px-4 py-2">Predicted</th>
                                                        <th className="px-4 py-2">Diff</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {forecastData.recent_performance.map((row, i) => (
                                                        <tr key={i} className="border-b border-border hover:bg-white/5">
                                                            <td className="px-4 py-2 text-text-primary">{row.date}</td>
                                                            <td className="px-4 py-2">{row.actual?.toFixed(2) ?? '-'}</td>
                                                            <td className="px-4 py-2">{row.predicted?.toFixed(2) ?? '-'}</td>
                                                            <td className={`px-4 py-2 font-medium ${Math.abs(row.actual - row.predicted) > 10 ? 'text-aqi-danger' : 'text-aqi-good'
                                                                }`}>
                                                                {row.actual ? (row.actual - row.predicted).toFixed(2) : '-'}
                                                            </td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                )}
                            </>
                        ) : (
                            !predicting && (
                                <div className="h-full flex flex-col items-center justify-center text-text-muted py-20">
                                    <FlaskConical size={48} className="opacity-20 mb-4" />
                                    <p>Select parameters and start a training run.</p>
                                </div>
                            )
                        )}
                        {predicting && <div className="h-[200px] flex items-center justify-center text-text-muted animate-pulse">Running Inference...</div>}
                    </div>
                )}

                {/* TAB CONTENT: TRENDS */}
                {activeTab === 'trends' && (
                    <div className="rounded-[20px] bg-dark-card border border-border p-6 shadow-sm min-h-[500px]">
                        <div className="flex flex-wrap gap-2 mb-6">
                            {POLLUTANTS.map(p => (
                                <button
                                    key={p}
                                    onClick={() => {
                                        if (selectedPollutants.includes(p)) {
                                            if (selectedPollutants.length > 1) setSelectedPollutants(prev => prev.filter(x => x !== p));
                                        } else {
                                            if (selectedPollutants.length < 3) setSelectedPollutants(prev => [...prev, p]);
                                        }
                                    }}
                                    className={`px-3 py-1 text-xs rounded-full border transition-colors ${selectedPollutants.includes(p)
                                            ? 'bg-secondary/10 border-secondary text-secondary'
                                            : 'bg-dark-base border-border text-text-muted hover:border-text-muted'
                                        }`}
                                >
                                    {p}
                                </button>
                            ))}
                        </div>

                        {trendsLoading ? (
                            <div className="h-[320px] flex items-center justify-center text-text-muted animate-pulse">Loading Trends...</div>
                        ) : (
                            <div className="h-[400px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={historyData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
                                        <XAxis
                                            dataKey="Date"
                                            tick={{ fill: '#8B949E', fontSize: 12 }}
                                            tickFormatter={(val) => new Date(val).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                                            axisLine={false} tickLine={false}
                                        />
                                        <YAxis tick={{ fill: '#8B949E', fontSize: 12 }} axisLine={false} tickLine={false} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#161B22', borderColor: '#30363d', color: '#E6EDF3', borderRadius: '8px' }}
                                            itemStyle={{ fontSize: '13px' }}
                                        />
                                        <Legend />
                                        {selectedPollutants.map((p, i) => (
                                            <Line
                                                key={p}
                                                type="monotone"
                                                dataKey={p}
                                                stroke={i === 0 ? "#2BD42E" : i === 1 ? "#3b82f6" : "#f59e0b"}
                                                strokeWidth={2}
                                                dot={false}
                                                activeDot={{ r: 6 }}
                                            />
                                        ))}
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        )}
                    </div>
                )}

                {/* TAB CONTENT: DISTRIBUTION */}
                {activeTab === 'distribution' && (
                    <div className="rounded-[20px] bg-dark-card border border-border p-6 shadow-sm min-h-[500px]">
                        <div className="flex items-center gap-4 mb-6">
                            <label className="text-sm text-text-muted">Select Metric:</label>
                            <select
                                value={distPollutant}
                                onChange={(e) => setDistPollutant(e.target.value)}
                                className="bg-dark-base border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary focus:outline-none focus:border-primary"
                            >
                                {POLLUTANTS.map(p => <option key={p} value={p}>{p}</option>)}
                            </select>
                        </div>

                        {distLoading ? (
                            <div className="h-[320px] flex items-center justify-center text-text-muted animate-pulse">Calculating Distribution...</div>
                        ) : (
                            <div className="h-[400px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={distData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
                                        <XAxis dataKey="range" tick={{ fill: '#8B949E', fontSize: 10 }} axisLine={false} tickLine={false} />
                                        <YAxis tick={{ fill: '#8B949E', fontSize: 12 }} axisLine={false} tickLine={false} />
                                        <Tooltip
                                            cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                            contentStyle={{ backgroundColor: '#161B22', borderColor: '#30363d', color: '#E6EDF3', borderRadius: '8px' }}
                                        />
                                        <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        )}
                    </div>
                )}

            </div>
        </div>
    );
};

export default ModelDashboard;
