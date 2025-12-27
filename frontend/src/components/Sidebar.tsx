import React from 'react';
import { LayoutDashboard, Database, TestTube2, CloudRain } from 'lucide-react';

interface SidebarProps {
    currentView: 'dashboard' | 'models' | 'data';
    setView: (view: 'dashboard' | 'models' | 'data') => void;
}

const Sidebar: React.FC<SidebarProps> = ({ currentView, setView }) => {
    const items = [
        { id: 'dashboard', label: 'AQI Dashboard', icon: LayoutDashboard },
        { id: 'models', label: 'Model Lab', icon: TestTube2 },
        { id: 'data', label: 'Data Ops', icon: Database },
    ];

    return (
        <div className="w-[80px] md:w-[240px] bg-dark-card border-r border-border h-screen sticky top-0 flex flex-col p-4">
            <div className="flex items-center gap-3 px-2 mb-8 mt-2">
                <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center text-primary">
                    <CloudRain size={20} />
                </div>
                <span className="text-lg font-bold text-white hidden md:block tracking-tight">ShuddhVayu</span>
            </div>

            <nav className="space-y-2 flex-1">
                {items.map(item => (
                    <button
                        key={item.id}
                        onClick={() => setView(item.id as any)}
                        className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-all ${currentView === item.id
                                ? 'bg-primary/10 text-primary font-medium'
                                : 'text-text-muted hover:text-text-primary hover:bg-dark-base'
                            }`}
                    >
                        <item.icon size={20} />
                        <span className="hidden md:block text-sm">{item.label}</span>
                    </button>
                ))}
            </nav>

            <div className="px-3 py-4 border-t border-border">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-secondary/20 flex items-center justify-center text-secondary text-xs font-bold">
                        CP
                    </div>
                    <div className="hidden md:flex flex-col">
                        <span className="text-sm text-text-primary font-medium">Admin User</span>
                        <span className="text-xs text-text-muted">Engineering</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Sidebar;
