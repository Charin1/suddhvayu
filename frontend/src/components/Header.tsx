import React from 'react';
import { Bell, Search, Calendar, User } from 'lucide-react';

const Header: React.FC = () => {
    return (
        <header className="h-[72px] bg-dark-card border-b border-border flex items-center justify-between px-6 md:px-8 sticky top-0 z-40">
            {/* Search Bar */}
            <div className="hidden md:flex items-center relative w-[320px]">
                <Search className="absolute left-3 text-text-muted w-4 h-4" />
                <input
                    type="text"
                    placeholder="Search metrics, cities, or models..."
                    className="w-full bg-dark-base border border-border rounded-xl pl-10 pr-4 py-2.5 text-sm text-text-primary focus:outline-none focus:border-primary transition-all placeholder:text-text-muted"
                />
            </div>

            {/* Right Actions */}
            <div className="flex items-center gap-4 ml-auto">
                <div className="flex items-center gap-2 px-3 py-1.5 bg-dark-base rounded-lg border border-border">
                    <Calendar size={14} className="text-text-muted" />
                    <span className="text-xs font-medium text-text-primary">
                        {new Date().toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })}
                    </span>
                </div>

                <div className="h-8 w-[1px] bg-border mx-2 hidden md:block" />

                <button className="relative p-2 text-text-muted hover:text-text-primary hover:bg-dark-base rounded-lg transition-colors">
                    <Bell size={20} />
                    <span className="absolute top-2 right-2 w-2 h-2 bg-aqi-danger rounded-full pointer-events-none" />
                </button>

                <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary shadow-sm ring-2 ring-transparent hover:ring-primary/20 cursor-pointer transition-all">
                    <User size={16} />
                </div>
            </div>
        </header>
    );
};

export default Header;
