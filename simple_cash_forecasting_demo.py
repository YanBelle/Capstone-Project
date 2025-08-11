#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Cash Forecasting Demo - Python 2.7 Compatible
====================================================

Demonstrates cash forecasting concepts using basic Python libraries
"""

import sys
import os
import random
import math
from datetime import datetime, timedelta

def generate_simple_forecast_data():
    """Generate simple synthetic data for demonstration"""
    print("Generating synthetic ATM transaction data...")
    
    # Create data for 3 terminals over 30 days
    terminals = [101, 102, 103]
    data = []
    
    for terminal_id in terminals:
        base_cash = 80000 if terminal_id == 101 else 60000
        current_cash = base_cash
        
        for day in range(30):
            # Simulate daily patterns
            daily_transactions = random.randint(30, 60)
            
            for _ in range(daily_transactions):
                if current_cash <= 5000:  # Refill
                    current_cash = base_cash
                
                # Random withdrawal
                withdrawal = random.choice([20, 40, 60, 80, 100, 120, 160, 200])
                
                if withdrawal <= current_cash:
                    current_cash -= withdrawal
                    
                    data.append({
                        'terminal_id': terminal_id,
                        'day': day,
                        'withdrawal': withdrawal,
                        'remaining_cash': current_cash
                    })
    
    print("Generated {} transaction records for {} terminals".format(len(data), len(terminals)))
    return data

def simple_linear_forecast(cash_history, days_ahead=7):
    """Simple linear regression forecast"""
    if len(cash_history) < 5:
        return None
    
    # Calculate trend (simple slope)
    n = len(cash_history)
    x_sum = sum(range(n))
    y_sum = sum(cash_history)
    xy_sum = sum(i * cash_history[i] for i in range(n))
    x2_sum = sum(i * i for i in range(n))
    
    # Linear regression: y = mx + b
    m = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
    b = (y_sum - m * x_sum) / n
    
    # Predict future values
    forecast = []
    for day in range(n, n + days_ahead):
        predicted_cash = m * day + b
        forecast.append(max(0, predicted_cash))  # Cash can't be negative
    
    return forecast

def moving_average_forecast(cash_history, window=7, days_ahead=7):
    """Moving average forecast"""
    if len(cash_history) < window:
        return None
    
    # Calculate moving average trend
    recent_avg = sum(cash_history[-window:]) / window
    
    # Simple trend calculation
    if len(cash_history) >= 2 * window:
        older_avg = sum(cash_history[-2*window:-window]) / window
        trend = (recent_avg - older_avg) / window
    else:
        trend = 0
    
    # Forecast
    forecast = []
    for day in range(days_ahead):
        predicted_cash = recent_avg + trend * day
        forecast.append(max(0, predicted_cash))
    
    return forecast

def calculate_depletion_risk(current_cash, daily_avg_withdrawal):
    """Calculate risk of cash depletion"""
    if daily_avg_withdrawal <= 0:
        return "Low", 999
    
    days_until_empty = current_cash / daily_avg_withdrawal
    
    if days_until_empty <= 2:
        risk = "High"
    elif days_until_empty <= 5:
        risk = "Medium"
    else:
        risk = "Low"
    
    return risk, days_until_empty

def create_simple_visualization(terminal_data):
    """Create a simple text-based visualization"""
    print("\n" + "="*60)
    print("CASH FORECASTING RESULTS")
    print("="*60)
    
    for terminal_id, data in terminal_data.items():
        cash_levels = [record['remaining_cash'] for record in data[-20:]]  # Last 20 transactions
        withdrawals = [record['withdrawal'] for record in data[-20:]]
        
        if not cash_levels:
            continue
        
        # Current status
        current_cash = cash_levels[-1]
        avg_withdrawal = sum(withdrawals) / len(withdrawals)
        
        # Forecasts
        linear_forecast = simple_linear_forecast(cash_levels, 7)
        ma_forecast = moving_average_forecast(cash_levels, 5, 7)
        
        # Risk assessment
        risk, days_until_empty = calculate_depletion_risk(current_cash, avg_withdrawal * 50)  # 50 transactions/day estimate
        
        print("\nTerminal {}:".format(terminal_id))
        print("  Current Cash: ${:,}".format(current_cash))
        print("  Average Withdrawal: ${:.2f}".format(avg_withdrawal))
        print("  Risk Level: {}".format(risk))
        print("  Days Until Depletion: {:.1f}".format(days_until_empty))
        
        if linear_forecast:
            print("  Linear Forecast (7 days): ${:,.0f} -> ${:,.0f}".format(
                linear_forecast[0], linear_forecast[-1]))
        
        if ma_forecast:
            print("  Moving Avg Forecast (7 days): ${:,.0f} -> ${:,.0f}".format(
                ma_forecast[0], ma_forecast[-1]))
        
        # Simple trend visualization
        trend_line = "  Cash Trend (last 10): "
        recent_levels = cash_levels[-10:]
        for i in range(len(recent_levels)-1):
            if recent_levels[i+1] > recent_levels[i]:
                trend_line += "^"
            elif recent_levels[i+1] < recent_levels[i]:
                trend_line += "v"
            else:
                trend_line += "-"
        print(trend_line)

def main():
    """Main function for simple cash forecasting demo"""
    print("Simple ATM Cash Forecasting System")
    print("=" * 50)
    print("Python Version: {}".format(sys.version))
    print("Compatible with Python 2.7")
    print()
    
    # Generate synthetic data
    data = generate_simple_forecast_data()
    
    # Group by terminal
    terminal_data = {}
    for record in data:
        terminal_id = record['terminal_id']
        if terminal_id not in terminal_data:
            terminal_data[terminal_id] = []
        terminal_data[terminal_id].append(record)
    
    # Create visualizations
    create_simple_visualization(terminal_data)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    total_transactions = len(data)
    total_withdrawals = sum(record['withdrawal'] for record in data)
    avg_withdrawal = total_withdrawals / total_transactions if total_transactions > 0 else 0
    
    print("Total Transactions: {:,}".format(total_transactions))
    print("Total Cash Dispensed: ${:,}".format(total_withdrawals))
    print("Average Withdrawal: ${:.2f}".format(avg_withdrawal))
    
    # Risk summary
    high_risk_terminals = 0
    for terminal_id, records in terminal_data.items():
        if records:
            current_cash = records[-1]['remaining_cash']
            recent_withdrawals = [r['withdrawal'] for r in records[-10:]]
            avg_daily = sum(recent_withdrawals) * 5  # Estimate daily usage
            
            if current_cash / avg_daily <= 2:
                high_risk_terminals += 1
    
    print("High Risk Terminals: {} / {}".format(high_risk_terminals, len(terminal_data)))
    
    print("\n" + "="*60)
    print("FORECASTING INSIGHTS")
    print("="*60)
    print("- Linear forecasting captures overall trends")
    print("- Moving averages smooth out daily variations")
    print("- Risk assessment helps prioritize refill scheduling")
    print("- Weekend patterns may require separate modeling")
    print("- Real-world implementation should include:")
    print("  - External factors (holidays, events)")
    print("  - Machine learning for complex patterns")
    print("  - Real-time monitoring and alerts")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
