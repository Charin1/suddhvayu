import React, { useEffect, useState } from 'react';
import { Database, FileText, CheckCircle2, XCircle, Clock, Save } from 'lucide-react';

interface FileInfo {
    key: string;
    category: 'raw' | 'processed';
    exists: boolean;
    path: string;
    size_bytes: number;
    last_modified: string | null;
}

interface DataPipelineStatusProps {
    searchTerm?: string;
}

const DataPipelineStatus: React.FC<DataPipelineStatusProps> = ({ searchTerm = "" }) => {
    const [files, setFiles] = useState<FileInfo[]>([]);
    const [loading, setLoading] = useState(true);

    const fetchStatus = async () => {
        try {
            const res = await fetch('/api/v1/system/status');
            if (res.ok) {
                const data = await res.json();
                // sort by last modified descending
                data.sort((a: FileInfo, b: FileInfo) => {
                    if (!a.last_modified) return 1;
                    if (!b.last_modified) return -1;
                    return new Date(b.last_modified).getTime() - new Date(a.last_modified).getTime();
                });
                setFiles(data);
            }
        } catch (e) {
            console.error("Failed to fetch system status", e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchStatus();
        const interval = setInterval(fetchStatus, 5000); // Poll every 5s
        return () => clearInterval(interval);
    }, []);

    const filteredFiles = files.filter(f => f.key.toLowerCase().includes(searchTerm.toLowerCase()));

    const formatBytes = (bytes: number) => {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    const formatDate = (dateStr: string | null) => {
        if (!dateStr) return 'Never';
        return new Date(dateStr).toLocaleString();
    };

    const renderFileRow = (file: FileInfo) => (
        <div key={file.path} className="flex items-center justify-between p-3 bg-dark-base rounded-xl border border-border group hover:border-primary/30 transition-colors gap-3">
            <div className="flex items-center gap-3 min-w-0 flex-1">
                <div className={`p-2 rounded-lg flex-shrink-0 ${file.exists ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}`}>
                    {file.exists ? <FileText size={16} /> : <XCircle size={16} />}
                </div>
                <div className="min-w-0 flex-1">
                    <div className="font-medium text-text-primary text-sm flex items-center gap-2">
                        <span className="truncate" title={file.key}>{file.key}</span>
                        {file.category === 'processed' && <span className="flex-shrink-0 text-[10px] bg-primary/20 text-primary px-1.5 py-0.5 rounded">PROCESSED</span>}
                    </div>
                    <div className="text-xs text-text-muted flex items-center gap-2 mt-0.5">
                        <span className="flex items-center gap-1 flex-shrink-0"><Save size={10} /> {formatBytes(file.size_bytes)}</span>
                        <span>•</span>
                        <span className="flex items-center gap-1 truncate"><Clock size={10} /> {formatDate(file.last_modified)}</span>
                    </div>
                </div>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
                {file.exists ?
                    <div className="flex items-center gap-1 text-green-500 text-xs font-medium bg-green-500/10 px-2 py-1 rounded-full">
                        <CheckCircle2 size={12} /> Ready
                    </div>
                    :
                    <div className="flex items-center gap-1 text-red-500 text-xs font-medium bg-red-500/10 px-2 py-1 rounded-full">
                        Missing
                    </div>
                }
            </div>
        </div>
    );

    return (
        <div className="rounded-[20px] bg-dark-card border border-border p-6 shadow-sm h-full">
            <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-aqi-good/10 rounded-lg text-aqi-good border border-aqi-good/20">
                    <Database size={20} />
                </div>
                <div>
                    <h3 className="text-lg font-semibold text-text-primary">Data Pipeline Overview</h3>
                    <p className="text-xs text-text-muted">Real-time status of raw & processed assets</p>
                </div>
            </div>

            <div className="space-y-3 max-h-[300px] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-border">
                {loading ? (
                    <div className="text-center text-text-muted text-xs py-4">Loading file status...</div>
                ) : filteredFiles.length > 0 ? (
                    filteredFiles.map(renderFileRow)
                ) : (
                    <div className="text-center text-text-muted text-xs py-4">
                        {searchTerm ? "No matching files found." : "No data files, run fetch first."}
                    </div>
                )}
            </div>
        </div>
    );
};

export default DataPipelineStatus;
