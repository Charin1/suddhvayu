import React, { useState } from 'react';
import { CloudRain, Play, CheckCircle2, Factory, Calendar, MapPin, Search, Building2 } from 'lucide-react';
import DataPipelineStatus from './DataPipelineStatus';
import { useLocation, AVAILABLE_STATES } from '../context/LocationContext';

const DataConsole: React.FC = () => {
    const {
        selectedState,
        selectedCity,
        availableCities,
        startDate,
        endDate,
        setSelectedState,
        setSelectedCity,
        setStartDate,
        setEndDate
    } = useLocation();

    // Weather Fetch State
    const [fetchingWeather, setFetchingWeather] = useState(false);
    const [weatherMsg, setWeatherMsg] = useState<string | null>(null);

    // Feature Processing State
    const [processingFeatures, setProcessingFeatures] = useState(false);
    const [featureMsg, setFeatureMsg] = useState<string | null>(null);

    // Search State
    const [searchTerm, setSearchTerm] = useState("");

    const handleFetchWeather = async () => {
        setFetchingWeather(true);
        setWeatherMsg(null);
        try {
            const res = await fetch('/api/v1/data/fetch-weather', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ city: selectedCity, start_date: startDate, end_date: endDate })
            });
            if (res.ok) {
                setWeatherMsg("✅ Fetch initiated! Watch the 'Data Pipeline Overview' panel for the new file to appear.");
            } else {
                setWeatherMsg(`❌ Failed: ${res.status} ${res.statusText}`);
            }
        } catch (e: any) {
            setWeatherMsg(`❌ Network error: ${e.message}`);
        } finally {
            setFetchingWeather(false);
            setTimeout(() => setWeatherMsg(null), 10000);
        }
    };

    const handleProcessFeatures = async () => {
        setProcessingFeatures(true);
        setFeatureMsg(null);
        try {
            const res = await fetch('/api/v1/data/process-features', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ city: selectedCity })
            });
            if (res.ok) {
                setFeatureMsg("✅ Pipeline started! Check 'processed' files in the Status Panel.");
            } else {
                setFeatureMsg(`❌ Failed: ${res.status} ${res.statusText}`);
            }
        } catch (e: any) {
            setFeatureMsg(`❌ Network error: ${e.message}`);
        } finally {
            setProcessingFeatures(false);
            setTimeout(() => setFeatureMsg(null), 10000);
        }
    };

    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

            {/* LEFT COLUMN: CONTROLS & LOGS */}
            <div className="md:col-span-2 space-y-6">

                {/* RAW DATA INGESTION CARD */}
                <div className="rounded-[20px] bg-dark-card border border-border p-6 shadow-sm">
                    <div className="flex items-center gap-3 mb-6">
                        <div className="p-2 bg-secondary/10 rounded-lg text-secondary border border-secondary/20">
                            <CloudRain size={20} />
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-text-primary">External Data Ingestion</h3>
                            <p className="text-xs text-text-muted">Fetch raw data from external APIs (Open-Meteo)</p>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 mb-4">
                        <div className="space-y-1">
                            <label className="text-xs font-medium text-text-muted flex items-center gap-1">
                                <MapPin size={10} /> State
                            </label>
                            <select
                                value={selectedState}
                                onChange={(e) => setSelectedState(e.target.value)}
                                className="w-full bg-dark-base border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-primary"
                            >
                                {AVAILABLE_STATES.map(s => <option key={s} value={s}>{s}</option>)}
                            </select>
                        </div>
                        <div className="space-y-1">
                            <label className="text-xs font-medium text-text-muted flex items-center gap-1">
                                <Building2 size={10} /> City
                            </label>
                            <select
                                value={selectedCity}
                                onChange={(e) => setSelectedCity(e.target.value)}
                                className="w-full bg-dark-base border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-primary"
                            >
                                {availableCities.map(c => <option key={c} value={c}>{c}</option>)}
                            </select>
                        </div>
                        <div className="space-y-1">
                            <label className="text-xs font-medium text-text-muted flex items-center gap-1">
                                <Calendar size={10} /> Start Date
                            </label>
                            <input
                                type="date"
                                value={startDate}
                                onChange={(e) => setStartDate(e.target.value)}
                                className="w-full bg-dark-base border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-primary"
                            />
                        </div>
                        <div className="space-y-1">
                            <label className="text-xs font-medium text-text-muted flex items-center gap-1">
                                <Calendar size={10} /> End Date
                            </label>
                            <input
                                type="date"
                                value={endDate}
                                onChange={(e) => setEndDate(e.target.value)}
                                className="w-full bg-dark-base border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-primary"
                            />
                        </div>
                    </div>

                    <div className="flex items-center justify-between p-4 bg-dark-base rounded-xl border border-border">
                        <div>
                            <div className="font-medium text-text-primary text-sm">Fetch Weather History</div>
                            <div className="text-xs text-text-muted mt-1">Triggers python script to pull daily metrics for {selectedCity}</div>
                        </div>
                        <button
                            onClick={() => {
                                console.log("Fetch button clicked");
                                handleFetchWeather();
                            }}
                            disabled={fetchingWeather}
                            className={`bg-secondary hover:bg-secondary/90 text-dark-base px-5 py-2 rounded-lg text-xs font-bold flex items-center gap-2 transition-colors ${fetchingWeather ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                            {fetchingWeather ? <CheckCircle2 className="animate-spin w-3 h-3" /> : <CloudRain className="w-3 h-3" />}
                            Fetch Data
                        </button>
                    </div>
                    {weatherMsg && (
                        <div className={`mt-3 text-xs p-2 rounded-lg border ${weatherMsg.includes('✅') ? 'text-green-400 bg-green-500/10 border-green-500/20' : 'text-red-400 bg-red-500/10 border-red-500/20'}`}>
                            {weatherMsg}
                        </div>
                    )}
                </div>

                {/* FEATURE ENGINEERING CARD */}
                <div className="rounded-[20px] bg-dark-card border border-border p-6 shadow-sm">
                    <div className="flex items-center gap-3 mb-6">
                        <div className="p-2 bg-primary/10 rounded-lg text-primary border border-primary/20">
                            <Factory size={20} />
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-text-primary">Feature Engineering Pipeline</h3>
                            <p className="text-xs text-text-muted">Transform raw data into model-ready features (lags, rolling stats)</p>
                        </div>
                    </div>

                    <div className="flex items-center justify-between p-4 bg-dark-base rounded-xl border border-border">
                        <div>
                            <div className="font-medium text-text-primary text-sm">Process Features for {selectedCity}</div>
                            <div className="text-xs text-text-muted mt-1">
                                Merges AQI + Weather, imputes missing values, creates lag/rolling features.
                                <br />Updates <code>{selectedCity || 'gujarat'}_features_for_model.csv</code>
                            </div>
                        </div>
                        <button
                            onClick={handleProcessFeatures}
                            disabled={processingFeatures}
                            className="bg-primary hover:bg-primary/90 text-dark-base px-5 py-2 rounded-lg text-xs font-bold flex items-center gap-2 transition-colors disabled:opacity-50"
                        >
                            {processingFeatures ? <CheckCircle2 className="animate-spin w-3 h-3" /> : <Play className="w-3 h-3" />}
                            Run Pipeline
                        </button>
                    </div>
                    {featureMsg && (
                        <div className={`mt-3 text-xs p-2 rounded-lg border ${featureMsg.includes('✅') ? 'text-primary bg-primary/10 border-primary/20' : 'text-red-400 bg-red-500/10 border-red-500/20'}`}>
                            {featureMsg}
                        </div>
                    )}
                </div>

            </div>

            {/* RIGHT COLUMN: STATUS & FILES */}
            <div className="md:col-span-1 space-y-6">
                <div className="relative">
                    <Search className="absolute left-3 top-2.5 text-text-muted w-4 h-4" />
                    <input
                        type="text"
                        placeholder="Search logs/files..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full bg-dark-card border border-border rounded-xl pl-10 pr-4 py-2 text-sm text-text-primary focus:outline-none focus:border-primary transition-all placeholder:text-text-muted shadow-sm"
                    />
                </div>

                <DataPipelineStatus searchTerm={searchTerm} />

                <div className="rounded-[20px] bg-dark-card border border-border p-5 shadow-sm">
                    <h4 className="text-sm font-semibold text-text-primary mb-3">Current Selection</h4>
                    <div className="space-y-2 text-xs text-text-muted">
                        <div className="flex items-center justify-between p-2 bg-dark-base rounded border border-border">
                            <span>State</span>
                            <span className="text-text-primary font-medium">{selectedState}</span>
                        </div>
                        <div className="flex items-center justify-between p-2 bg-dark-base rounded border border-border">
                            <span>City</span>
                            <span className="text-text-primary font-medium">{selectedCity}</span>
                        </div>
                        <div className="flex items-center justify-between p-2 bg-dark-base rounded border border-border">
                            <span>Date Range</span>
                            <span className="text-text-primary font-medium text-[10px]">{startDate} → {endDate}</span>
                        </div>
                    </div>
                </div>
            </div>

        </div>
    );
};

export default DataConsole;
