import React, { useState, useEffect } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Area, AreaChart
} from 'recharts';
import { Activity } from 'lucide-react';

interface ForecastResponse {
    forecast: { date: string; value: number }[];
    metadata: { mae: number; last_trained: string };
}

const ForecastViewer: React.FC = () => {
    const [historyData, setHistoryData] = useState<any[]>([]);
    const [forecastData, setForecastData] = useState<ForecastResponse | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchAll = async () => {
            try {
                // 1. History
                const histRes = await fetch('/api/v1/history/Ahmedabad');
                if (histRes.ok) {
                    const json = await histRes.json();
                    setHistoryData(json.data);
                }

                // 2. Default Forecast (XGBoost, PM2.5) for display
                const predRes = await fetch('/api/v1/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_type: "XGBoost", city: "Ahmedabad", days: 7, target: "PM2.5" })
                });
                if (predRes.ok) {
                    const predJson = await predRes.json();
                    setForecastData(predJson);
                }

            } catch (e) {
                console.error("Failed to load dashboard data", e);
            } finally {
                setLoading(false);
            }
        };
        fetchAll();
    }, []);

    if (loading) return <div className="h-[300px] flex items-center justify-center text-text-muted animate-pulse">Loading Intelligence...</div>;

    return (
        <div className="space-y-6">
            {/* Trends Card */}
            <div className="rounded-[20px] bg-dark-card border border-border p-6 shadow-sm">
                <div className="flex items-center justify-between mb-6">
                    <div>
                        <h3 className="text-xl font-semibold text-text-primary tracking-tight">Recent Trends</h3>
                        <p className="text-sm text-text-muted mt-1">Air Quality History (Last 30 Days)</p>
                    </div>
                    <Activity className="text-primary opacity-50" />
                </div>
                <div className="h-[250px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={historyData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
                            <XAxis
                                dataKey="Date"
                                tick={{ fill: '#8B949E', fontSize: 10 }}
                                tickFormatter={(val) => new Date(val).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                                axisLine={false} tickLine={false}
                            />
                            <YAxis tick={{ fill: '#8B949E', fontSize: 12 }} axisLine={false} tickLine={false} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#161B22', borderColor: '#30363d', color: '#E6EDF3', borderRadius: '8px' }}
                            />
                            <Line type="monotone" dataKey="AQI" stroke="#2BD42E" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="PM2.5" stroke="#3b82f6" strokeWidth={2} dot={false} strokeDasharray="5 5" />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Forecast Card */}
            <div className="rounded-[20px] bg-dark-card border border-border p-6 shadow-sm">
                <div className="flex items-center justify-between mb-6">
                    <div>
                        <h3 className="text-xl font-semibold text-text-primary tracking-tight">7-Day Forecast (PM2.5)</h3>
                        <p className="text-sm text-text-muted mt-1">AI-Powered Prediction (XGBoost)</p>
                    </div>
                    <div className="flex items-center gap-2 text-xs bg-dark-base px-2 py-1 rounded-lg border border-border">
                        <span className="text-text-muted">MAE:</span>
                        <span className="text-primary font-bold">{forecastData?.metadata.mae?.toFixed(2) || 'N/A'}</span>
                    </div>
                </div>

                {forecastData ? (
                    <div className="h-[250px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={forecastData.forecast}>
                                <defs>
                                    <linearGradient id="colorValueForecast" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#2BD42E" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#2BD42E" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
                                <XAxis dataKey="date" tick={{ fill: '#8B949E', fontSize: 10 }} tickFormatter={(val) => new Date(val).toLocaleDateString(undefined, { weekday: 'short' })} axisLine={false} tickLine={false} />
                                <YAxis tick={{ fill: '#8B949E', fontSize: 12 }} axisLine={false} tickLine={false} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#161B22', borderColor: '#30363d', color: '#E6EDF3', borderRadius: '8px' }}
                                />
                                <Area type="monotone" dataKey="value" stroke="#2BD42E" fill="url(#colorValueForecast)" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                ) : (
                    <div className="h-[250px] flex items-center justify-center text-text-muted">No forecast data</div>
                )}
            </div>
        </div>
    );
};

export default ForecastViewer;
