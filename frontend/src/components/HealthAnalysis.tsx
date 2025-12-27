import React, { useState } from 'react';
import { Activity, Shield, AlertTriangle, User } from 'lucide-react';

interface HealthResponse {
    summary: string;
    risk_level: string;
    recommendations: {
        children: string;
        seniors: string;
        athletes: string;
    };
}

const HealthAnalysis: React.FC = () => {
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState<HealthResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    const fetchAnalysis = async () => {
        setLoading(true);
        setError(null);
        try {
            // 1. Fetch Current Condition
            let currentAqi = 150; // Fallback
            let dominant = "PM2.5";

            try {
                const histRes = await fetch('/api/v1/history/Ahmedabad');
                if (histRes.ok) {
                    const histJson = await histRes.json();
                    if (histJson.data && histJson.data.length > 0) {
                        const latest = histJson.data[histJson.data.length - 1];
                        currentAqi = latest.AQI || 150;
                    }
                }
            } catch (e) {
                console.warn("Failed to fetch history for health analysis context", e);
            }

            // 2. Fetch Forecast (for tomorrow)
            let predictedAqi = currentAqi;
            try {
                const predRes = await fetch('/api/v1/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_type: "XGBoost", city: "Ahmedabad", days: 1, target: "AQI" })
                });
                if (predRes.ok) {
                    const predJson = await predRes.json();
                    if (predJson.forecast && predJson.forecast.length > 0) {
                        predictedAqi = predJson.forecast[0].value;
                    }
                }
            } catch (e) {
                console.warn("Failed to fetch forecast for health analysis context", e);
            }

            const payload = {
                current_aqi: Math.round(currentAqi),
                predicted_aqi: Math.round(predictedAqi),
                dominant_pollutant: dominant
            };

            const response = await fetch('/api/v1/analyze-health', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error('Failed to fetch analysis');

            const result = await response.json();
            setData(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="rounded-[20px] bg-dark-card border border-border overflow-hidden h-full">
            <div className="p-6 border-b border-border flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-primary/10 rounded-xl text-primary border border-primary/20">
                        <Activity className="w-5 h-5" />
                    </div>
                    <div>
                        <h3 className="font-semibold text-lg text-text-primary">AI Health Impact</h3>
                        <p className="text-xs text-text-muted">Real-time advisory</p>
                    </div>
                </div>
                {!data && !loading && (
                    <button
                        onClick={fetchAnalysis}
                        className="text-xs bg-primary/10 hover:bg-primary/20 text-primary px-3 py-1.5 rounded-full font-medium transition-colors"
                    >
                        Generate
                    </button>
                )}
            </div>

            <div className="p-6">
                {!data && !loading && !error && (
                    <div className="flex flex-col items-center justify-center py-12 text-center text-text-muted">
                        <Shield className="w-12 h-12 mb-3 opacity-20" />
                        <p className="text-sm">Click "Generate" to see AI-powered health recommendations based on current air quality.</p>
                    </div>
                )}

                {loading && (
                    <div className="space-y-4 animate-pulse py-4">
                        <div className="h-4 bg-dark-base rounded w-3/4"></div>
                        <div className="h-4 bg-dark-base rounded w-1/2"></div>
                        <div className="h-24 bg-dark-base rounded w-full mt-6"></div>
                    </div>
                )}

                {error && (
                    <div className="p-4 bg-aqi-danger/10 border border-aqi-danger/20 text-aqi-danger rounded-xl flex items-center gap-3 text-sm">
                        <AlertTriangle className="w-5 h-5 shrink-0" />
                        {error}
                    </div>
                )}

                {data && (
                    <div className="space-y-6">
                        {/* Risk Level Badge */}
                        <div className="flex items-center justify-between bg-dark-base p-4 rounded-xl border border-border">
                            <span className="text-sm text-text-muted font-medium">Risk Level</span>
                            <span className={`text-sm font-bold px-3 py-1 rounded-full border ${data.risk_level.toLowerCase().includes('high') ? 'bg-aqi-danger/10 text-aqi-danger border-aqi-danger/20' :
                                data.risk_level.toLowerCase().includes('moderate') ? 'bg-aqi-warning/10 text-aqi-warning border-aqi-warning/20' :
                                    'bg-aqi-good/10 text-aqi-good border-aqi-good/20'
                                }`}>
                                {data.risk_level.toUpperCase()}
                            </span>
                        </div>

                        <div className="space-y-2">
                            <p className="text-sm text-text-primary leading-relaxed">
                                {data.summary}
                            </p>
                        </div>

                        <div className="space-y-4">
                            <h4 className="text-xs font-semibold uppercase tracking-wider text-text-muted mb-3">Recommendations</h4>

                            <div className="space-y-3">
                                <div className="bg-dark-base border border-border p-3 rounded-xl">
                                    <div className="flex items-center gap-2 mb-1.5">
                                        <User className="w-3.5 h-3.5 text-primary" />
                                        <span className="text-xs font-bold text-text-primary">Children</span>
                                    </div>
                                    <p className="text-xs text-text-muted leading-snug">{data.recommendations.children}</p>
                                </div>
                                <div className="bg-dark-base border border-border p-3 rounded-xl">
                                    <div className="flex items-center gap-2 mb-1.5">
                                        <User className="w-3.5 h-3.5 text-primary" />
                                        <span className="text-xs font-bold text-text-primary">Seniors</span>
                                    </div>
                                    <p className="text-xs text-text-muted leading-snug">{data.recommendations.seniors}</p>
                                </div>
                                <div className="bg-dark-base border border-border p-3 rounded-xl">
                                    <div className="flex items-center gap-2 mb-1.5">
                                        <Activity className="w-3.5 h-3.5 text-primary" />
                                        <span className="text-xs font-bold text-text-primary">Athletes</span>
                                    </div>
                                    <p className="text-xs text-text-muted leading-snug">{data.recommendations.athletes}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default HealthAnalysis;
