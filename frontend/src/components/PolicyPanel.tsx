import React, { useState } from 'react';
import { ArrowRight, Activity, ShieldCheck, Zap } from 'lucide-react';

interface PolicyPlan {
    duration: string;
    focus: string;
    actions: string[];
    projected_impact: string;
}

interface PolicyData {
    short_term: PolicyPlan;
    medium_term: PolicyPlan;
    long_term: PolicyPlan;
}

interface PolicyPanelProps {
    city: string;
}

const PolicyPanel: React.FC<PolicyPanelProps> = ({ city }) => {
    const [data, setData] = useState<PolicyData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleGenerate = async () => {
        setLoading(true);
        setError('');
        try {
            // 1. Fetch Context Data (Current AQI)
            let currentAqi = 150;
            let dominantPollutant = "PM2.5";

            try {
                const histRes = await fetch(`/api/v1/history/${city}`);
                if (histRes.ok) {
                    const histJson = await histRes.json();
                    if (histJson.data && histJson.data.length > 0) {
                        const latest = histJson.data[histJson.data.length - 1];
                        currentAqi = latest.AQI || 150;
                        // Simple dominant logic if not provided
                        if (latest.PM25 > 100) dominantPollutant = "PM2.5";
                        else if (latest.PM10 > 100) dominantPollutant = "PM10";
                    }
                }
            } catch (e) {
                console.warn("PolicyPanel: Failed to fetch history", e);
            }

            const response = await fetch('/api/v1/analyze-policy', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    city,
                    current_aqi: currentAqi,
                    dominant_pollutant: dominantPollutant,
                }),
            });

            if (!response.ok) throw new Error('Failed to generate policy');
            const result = await response.json();
            setData(result);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20 mt-6 shadow-xl">
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h2 className="text-2xl font-bold bg-gradient-to-r from-teal-400 to-emerald-400 bg-clip-text text-transparent flex items-center gap-2">
                        <ShieldCheck className="w-6 h-6 text-emerald-400" />
                        Government Policy Recommendation Engine
                    </h2>
                    <p className="text-gray-400 text-sm mt-1">AI-driven roadmap to reduce pollution in {city} under safety limits.</p>
                </div>
                <button
                    onClick={handleGenerate}
                    disabled={loading}
                    className="px-6 py-2 bg-gradient-to-r from-emerald-500 to-teal-600 rounded-lg font-semibold hover:shadow-lg hover:shadow-emerald-500/20 transition-all disabled:opacity-50 flex items-center gap-2"
                >
                    {loading ? 'Simulating Impact...' : 'Generate Policy Roadmap'}
                    {!loading && <ArrowRight className="w-4 h-4" />}
                </button>
            </div>

            {error && <div className="text-red-400 bg-red-400/10 p-4 rounded-xl mb-4">{error}</div>}

            {data && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
                    {/* Short Term */}
                    <PolicyCard
                        title="Short Term"
                        icon={<Zap className="w-5 h-5 text-yellow-400" />}
                        plan={data.short_term}
                        color="border-yellow-400/30 bg-yellow-400/5"
                        impactColor="text-yellow-400"
                    />

                    {/* Medium Term */}
                    <PolicyCard
                        title="Medium Term"
                        icon={<Activity className="w-5 h-5 text-blue-400" />}
                        plan={data.medium_term}
                        color="border-blue-400/30 bg-blue-400/5"
                        impactColor="text-blue-400"
                    />

                    {/* Long Term */}
                    <PolicyCard
                        title="Long Term"
                        icon={<ShieldCheck className="w-5 h-5 text-emerald-400" />}
                        plan={data.long_term}
                        color="border-emerald-400/30 bg-emerald-400/5"
                        impactColor="text-emerald-400"
                    />
                </div>
            )}

            {/* Simulation/Progress Bar Visualization */}
            {data && (
                <div className="mt-8 p-6 bg-black/20 rounded-xl border border-white/10">
                    <h3 className="text-lg font-semibold text-gray-200 mb-4 flex items-center gap-2">
                        <Activity className="w-5 h-5 text-purple-400" /> Projected Impact Simulation
                    </h3>
                    <div className="flex flex-col gap-4">
                        <ImpactBar label="Current AQI" value={100} color="bg-red-500" text="Base Level" />
                        <ImpactBar label="After 3 Months" value={85} color="bg-yellow-500" text="~15% Reduction" width="85%" />
                        <ImpactBar label="After 6 Months" value={70} color="bg-blue-500" text="~30% Reduction" width="70%" />
                        <ImpactBar label="After 12 Months" value={50} color="bg-emerald-500" text="Safety Level Achieved" width="50%" />
                    </div>
                </div>
            )}
        </div>
    );
};

// Helper Components
const PolicyCard = ({ title, icon, plan, color, impactColor }: { title: string, icon: any, plan: PolicyPlan, color: string, impactColor: string }) => (
    <div className={`p-5 rounded-xl border ${color} hover:scale-[1.02] transition-transform`}>
        <div className="flex items-center gap-2 mb-3">
            {icon}
            <h3 className="font-bold text-lg text-gray-100">{title}</h3>
        </div>
        <div className="text-xs font-mono text-gray-500 mb-2 uppercase tracking-wider">{plan.duration} • {plan.focus}</div>

        <ul className="space-y-2 mb-4">
            {plan.actions.map((action, i) => (
                <li key={i} className="text-sm text-gray-300 flex items-start gap-2">
                    <span className="text-emerald-500 mt-1">•</span>
                    {action}
                </li>
            ))}
        </ul>

        <div className={`text-xs font-semibold ${impactColor} border-t border-white/10 pt-3 mt-auto`}>
            IMPACT: {plan.projected_impact}
        </div>
    </div>
);

// Fixed: Explicit type for props
const ImpactBar = ({ label, value, color, text, width = '100%' }: { label: string, value: number, color: string, text: string, width?: string }) => (
    <div className="flex items-center gap-4">
        <div className="w-32 text-sm text-gray-400 text-right">{label}</div>
        <div className="flex-1 h-3 bg-gray-800 rounded-full overflow-hidden">
            <div className={`h-full ${color} rounded-full transition-all duration-1000 ease-out`} style={{ width: width }}></div>
        </div>
        <div className="w-40 text-xs font-mono text-gray-300">{text}</div>
    </div>
);

export default PolicyPanel;
