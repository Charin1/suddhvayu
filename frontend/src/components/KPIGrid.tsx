import React, { useEffect, useState } from 'react';
import { Activity, Wind, TrendingUp, TrendingDown, AlertCircle } from 'lucide-react';

interface KPICardProps {
    title: string;
    value: string | number;
    subtitle: string;
    icon: React.ReactNode;
    color?: 'primary' | 'warning' | 'danger' | 'good';
}

const KPICard: React.FC<KPICardProps> = ({ title, value, subtitle, icon, color = 'primary' }) => {
    const colorMap = {
        primary: 'text-primary border-primary/20',
        warning: 'text-aqi-warning border-aqi-warning/20',
        danger: 'text-aqi-danger border-aqi-danger/20',
        good: 'text-aqi-good border-aqi-good/20',
    };

    const activeColor = colorMap[color] || colorMap['primary'];

    return (
        <div className={`
      relative overflow-hidden rounded-[16px] bg-dark-card border border-border p-5
      transition-all duration-300 hover:shadow-lg hover:border-opacity-50
      w-full h-[130px] flex flex-col justify-between
    `}>
            <div className="flex justify-between items-start">
                <div className="flex flex-col">
                    <span className="text-text-primary/60 text-xs font-medium uppercase tracking-wider">{title}</span>
                    <span className={`text-2xl font-bold mt-1 ${activeColor.split(' ')[0]}`}>{value}</span>
                </div>
                <div className={`p-2 rounded-full bg-dark-base border ${activeColor}`}>
                    {React.cloneElement(icon as React.ReactElement, { size: 18 })}
                </div>
            </div>

            <div className="flex items-center gap-2 mt-2">
                <span className="text-text-muted text-xs font-medium">{subtitle}</span>
            </div>
        </div>
    );
};

const KPIGrid: React.FC = () => {
    const [stats, setStats] = useState({
        aqi: '...',
        aqiSubtitle: 'Loading...',
        aqiColor: 'primary',
        pollutant: '...',
        trendValue: '...',
        trendDirection: 'neutral',
        alertStatus: '...',
        alertSubtitle: '...'
    });

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await fetch('/api/v1/history/Ahmedabad');
                if (res.ok) {
                    const json = await res.json();
                    const data = json.data;
                    if (data && data.length > 0) {
                        const latest = data[data.length - 1];
                        const prev = data.length > 1 ? data[data.length - 2] : latest;

                        // AQI Logic
                        const aqi = latest.AQI || 0;
                        let aqiColor = 'good';
                        let aqiSubtitle = 'Good';
                        if (aqi > 50) { aqiColor = 'good'; aqiSubtitle = 'Moderate'; }
                        if (aqi > 100) { aqiColor = 'warning'; aqiSubtitle = 'Unhealthy for Sensitive Groups'; }
                        if (aqi > 150) { aqiColor = 'danger'; aqiSubtitle = 'Unhealthy'; }
                        if (aqi > 200) { aqiColor = 'danger'; aqiSubtitle = 'Very Unhealthy'; }

                        // Trend Logic
                        const change = aqi - (prev.AQI || aqi);
                        const pctChange = ((change / (prev.AQI || 1)) * 100).toFixed(1);
                        const trendValue = `${change > 0 ? '+' : ''}${pctChange}%`;
                        const trendDirection = change > 0 ? 'up' : change < 0 ? 'down' : 'neutral';

                        // Dominant Pollutant (Simple heuristic: max of sub-indices, or just PM2.5 for now as we don't have sub-indices computed)
                        const pollutant = "PM 2.5"; // Usually the driver

                        setStats({
                            aqi: Math.round(aqi).toString(),
                            aqiSubtitle: aqiSubtitle,
                            aqiColor: aqiColor,
                            pollutant: pollutant,
                            trendValue: trendValue,
                            trendDirection: trendDirection,
                            alertStatus: aqiSubtitle,
                            alertSubtitle: aqi > 200 ? "Health Alert in Effect" : "No active health advisories"
                        });
                    }
                }
            } catch (e) {
                console.error("KPI fetch error", e);
            }
        };
        fetchData();
    }, []);

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 w-full">
            <KPICard
                title="Current AQI"
                value={stats.aqi}
                subtitle={stats.aqiSubtitle}
                icon={<Activity />}
                color={stats.aqiColor as any}
            />
            <KPICard
                title="Main Pollutant"
                value={stats.pollutant}
                subtitle="Dominant Concentration"
                icon={<Wind />}
                color="primary"
            />
            <KPICard
                title="24h Change"
                value={stats.trendValue}
                subtitle={stats.trendDirection === 'up' ? "Worsening Trend" : "Improving Trend"}
                icon={stats.trendDirection === 'up' ? <TrendingUp /> : <TrendingDown />}
                color={stats.trendDirection === 'up' ? 'danger' : 'good'}
            />
            <KPICard
                title="Alert Status"
                value={stats.alertStatus}
                subtitle={stats.alertSubtitle}
                icon={<AlertCircle />}
                color={stats.aqiColor === 'danger' ? 'danger' : 'good'}
            />
        </div>
    );
};

export default KPIGrid;
