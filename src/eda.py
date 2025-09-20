import pandas as pd
import plotly.express as px

def plot_trends(df, selected_pollutants, city):
    """Generates a line plot for comparing historical pollutant trends."""
    fig = px.line(
        df,
        x='Date',
        y=selected_pollutants,
        title=f"Historical Trends for {', '.join(selected_pollutants)} in {city}"
    )
    fig.update_layout(template="plotly_white", yaxis_title="Concentration / Index Value")
    return fig

def plot_distribution(df, pollutant, city):
    """Generates a box plot to show the yearly distribution of a pollutant."""
    df['Year'] = df['Date'].dt.year
    fig = px.box(
        df,
        x='Year',
        y=pollutant,
        title=f"Yearly Distribution of {pollutant} in {city}"
    )
    fig.update_layout(template="plotly_white")
    return fig

def plot_missing_values(raw_df, pollutants, city):
    """Generates a bar chart showing the number of missing values in the raw data."""
    df_city_raw = raw_df[raw_df['City'] == city]
    missing_values = df_city_raw[pollutants].isnull().sum().reset_index()
    missing_values.columns = ['Pollutant', 'Number of Missing Days']
    
    fig = px.bar(
        missing_values,
        x='Pollutant',
        y='Number of Missing Days',
        title=f"Total Missing Data Points (Days) in Raw Data for {city}"
    )
    return fig
