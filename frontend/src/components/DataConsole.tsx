import React, { useState } from 'react';
import { CloudRain, Play, CheckCircle2 } from 'lucide-react';
import DataPipelineStatus from './DataPipelineStatus';

const DataConsole: React.FC = () => {
    const [fetching, setFetching] = useState(false);
    const [message, setMessage] = useState<string | null>(null);

    const handleFetchWeather = async () => {
        setFetching(true);
        setMessage(null);
        try {
            const res = await fetch('/api/v1/data/fetch-weather', { method: 'POST' });
            if (res.ok) {
                setMessage("Weather data fetch initiated. Process running in background.");
            } else {
                setMessage("Failed to start data fetch.");
            }
        } catch (e) {
            setMessage("Network error.");
        } finally {
            setFetching(false);
        }
    };

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="md:col-span-2">
                <DataPipelineStatus />
            </div>

            <div className="rounded-[20px] bg-dark-card border border-border p-6 shadow-sm">
                <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 bg-secondary/10 rounded-lg text-secondary border border-secondary/20">
                        <CloudRain size={20} />
                    </div>
                    <div>
                        <h3 className="text-lg font-semibold text-text-primary">External Data Sources</h3>
                        <p className="text-xs text-text-muted">Manage integrations & cron jobs</p>
                    </div>
                </div>

                <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 bg-dark-base rounded-xl border border-border">
                        <div>
                            <div className="font-medium text-text-primary text-sm">Open-Meteo Weather API</div>
                            <div className="text-xs text-text-muted mt-1">Fetches historical weather for Ahmedabad (2015-2020)</div>
                        </div>
                        <button
                            onClick={handleFetchWeather}
                            disabled={fetching}
                            className="bg-primary hover:bg-primary/90 text-dark-base px-4 py-2 rounded-lg text-xs font-bold flex items-center gap-2 transition-colors disabled:opacity-50"
                        >
                            {fetching ? <CheckCircle2 className="animate-spin w-3 h-3" /> : <Play className="w-3 h-3" />}
                            Run Script
                        </button>
                    </div>

                    {message && (
                        <div className="text-xs text-primary bg-primary/10 p-2 rounded-lg border border-primary/20">
                            {message}
                        </div>
                    )}
                </div>
            </div>

            <div className="rounded-[20px] bg-dark-card border border-border p-6 shadow-sm flex items-center justify-center text-text-muted text-sm">
                More data connectors coming soon...
            </div>
        </div>
    );
};

export default DataConsole;
