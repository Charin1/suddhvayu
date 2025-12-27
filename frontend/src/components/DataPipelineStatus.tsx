import React, { useEffect, useState } from 'react';
import { Database, FileText, CheckCircle2, XCircle } from 'lucide-react';

interface FileInfo {
    exists: boolean;
    modified: string | null;
    size_kb: number;
}

interface SystemStatus {
    raw_data: Record<string, FileInfo>;
    processed_data: Record<string, FileInfo>;
}

const DataPipelineStatus: React.FC = () => {
    const [status, setStatus] = useState<SystemStatus | null>(null);
    const [loading, setLoading] = useState(true);

    const fetchStatus = async () => {
        setLoading(true);
        try {
            const res = await fetch('/api/v1/system/status');
            if (res.ok) {
                setStatus(await res.json());
            }
        } catch (e) {
            console.error("Status fetch failed", e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchStatus();
        const interval = setInterval(fetchStatus, 30000); // Poll every 30s
        return () => clearInterval(interval);
    }, []);

    const StatusRow = ({ name, info }: { name: string, info: FileInfo }) => (
        <div className="flex items-center justify-between py-3 border-b border-border last:border-0 hover:bg-white/5 px-2 rounded-lg transition-colors">
            <div className="flex items-center gap-3">
                <FileText className="text-text-muted w-4 h-4" />
                <div>
                    <div className="text-sm font-medium text-text-primary">{name}</div>
                    <div className="text-xs text-text-muted">
                        {info.exists ? `${info.size_kb} KB` : 'Missing'}
                    </div>
                </div>
            </div>
            <div className="flex flex-col items-end">
                {info.exists ? (
                    <div className="flex items-center gap-1.5 text-aqi-good text-xs font-semibold bg-aqi-good/10 px-2 py-1 rounded-full border border-aqi-good/20">
                        <CheckCircle2 size={12} /> Ready
                    </div>
                ) : (
                    <div className="flex items-center gap-1.5 text-aqi-danger text-xs font-semibold bg-aqi-danger/10 px-2 py-1 rounded-full border border-aqi-danger/20">
                        <XCircle size={12} /> Missing
                    </div>
                )}
                <div className="text-[10px] text-text-muted mt-1">{info.modified || '-'}</div>
            </div>
        </div>
    );

    if (loading && !status) return <div className="text-sm text-text-muted animate-pulse">Checking pipeline health...</div>;

    return (
        <div className="rounded-[20px] bg-dark-card border border-border p-6 shadow-sm h-full">
            <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-secondary/10 rounded-lg text-secondary border border-secondary/20">
                    <Database size={20} />
                </div>
                <div>
                    <h3 className="text-lg font-semibold text-text-primary">Data Pipeline Overview</h3>
                    <p className="text-xs text-text-muted">Monitor raw & processed asset health</p>
                </div>
            </div>

            {status && (
                <div className="space-y-6">
                    <div>
                        <h4 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2 pl-2">Raw Ingestion</h4>
                        <div className="space-y-1">
                            {Object.entries(status.raw_data).map(([name, info]) => (
                                <StatusRow key={name} name={name} info={info} />
                            ))}
                        </div>
                    </div>
                    <div>
                        <h4 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2 pl-2">Processed Features</h4>
                        <div className="space-y-1">
                            {Object.entries(status.processed_data).map(([name, info]) => (
                                <StatusRow key={name} name={name} info={info} />
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default DataPipelineStatus;
