import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import KPIGrid from './components/KPIGrid';
import HealthAnalysis from './components/HealthAnalysis';
import ForecastViewer from './components/ForecastViewer';
import ModelDashboard from './components/ModelDashboard';
import DataConsole from './components/DataConsole';

const App: React.FC = () => {
    const [currentView, setCurrentView] = useState<'dashboard' | 'models' | 'data'>('dashboard');

    return (
        <div className="flex min-h-screen bg-dark-base text-text-primary font-sans selection:bg-primary/20">
            {/* Sidebar Navigation */}
            <Sidebar currentView={currentView} setView={setCurrentView} />

            {/* Main Content Area */}
            <div className="flex-1 flex flex-col h-screen overflow-hidden">
                <Header />

                <main className="flex-1 overflow-y-auto p-4 md:p-8 scrollbar-hide">
                    <div className="max-w-7xl mx-auto space-y-8 pb-20">

                        {/* VIEW: AQI INTELLIGENCE (User Facing) */}
                        {currentView === 'dashboard' && (
                            <>
                                <div className="flex flex-col gap-2">
                                    <h1 className="text-3xl font-bold tracking-tight">Ahmedabad Air Intelligence</h1>
                                    <p className="text-text-muted">Real-time monitoring and AI-powered health situational awareness.</p>
                                </div>
                                <KPIGrid />
                                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                                    <div className="lg:col-span-2 space-y-8">
                                        <ForecastViewer />
                                    </div>
                                    <div className="lg:col-span-1">
                                        <HealthAnalysis />
                                    </div>
                                </div>
                            </>
                        )}

                        {/* VIEW: MODEL LAB (Dev / ML Ops) */}
                        {currentView === 'models' && (
                            <>
                                <div className="flex flex-col gap-2 mb-4">
                                    <h1 className="text-3xl font-bold tracking-tight text-white">Model Training Lab</h1>
                                    <p className="text-text-muted">Experimental environment for retraining models and analyzing feature drift.</p>
                                </div>
                                <ModelDashboard />
                            </>
                        )}

                        {/* VIEW: DATA OPS */}
                        {currentView === 'data' && (
                            <>
                                <div className="flex flex-col gap-2 mb-4">
                                    <h1 className="text-3xl font-bold tracking-tight text-white">Data Operations</h1>
                                    <p className="text-text-muted">Manage ingestion pipelines and external data sources.</p>
                                </div>
                                <DataConsole />
                            </>
                        )}

                    </div>
                </main>
            </div>
        </div>
    );
};

export default App;
