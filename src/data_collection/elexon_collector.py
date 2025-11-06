"""
Elexon BMRS Data Collection Module

Collects data from Elexon Balancing Mechanism Reporting Service including:
- System Buy/Sell Prices
- Imbalance Prices
- Balancing Mechanism actions
- Generation by fuel type
- Day-ahead prices
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import time
from loguru import logger

from ..utils import (
    load_config,
    setup_logging,
    save_dataframe,
    get_api_key,
    parse_date,
    generate_date_range
)


class ElexonBMRSCollector:
    """Collector for Elexon BMRS data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Elexon BMRS collector.
        
        Args:
            config: Configuration dictionary. If None, loads from config.yaml
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self.api_config = config['apis']['elexon']
        self.base_url = self.api_config['base_url']
        self.rate_limit = self.api_config['rate_limit']
        self.last_request_time = 0
        
        # Get API key from environment
        try:
            self.api_key = get_api_key('ELEXON')
        except ValueError as e:
            logger.warning(f"API key not found: {e}")
            self.api_key = None
        
        setup_logging(config)
        logger.info("Elexon BMRS Collector initialized")
    
    def _rate_limit_wait(self):
        """Implement rate limiting between requests."""
        min_interval = 60 / self.rate_limit
        elapsed = time.time() - self.last_request_time
        
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(
        self,
        report_type: str,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Make HTTP request to Elexon BMRS API with rate limiting.
        
        Args:
            report_type: BMRS report type code (e.g., 'B1770')
            params: Query parameters
            
        Returns:
            JSON response as dictionary
        """
        if self.api_key is None:
            raise ValueError("API key required for Elexon BMRS. Set ELEXON_API_KEY in .env file")
        
        self._rate_limit_wait()
        
        # Build URL
        url = f"{self.base_url}/{report_type}/v1"
        
        # Add API key to params
        if params is None:
            params = {}
        params['APIKey'] = self.api_key
        params['ServiceType'] = 'json'
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {report_type}: {e}")
            raise
    
    def collect_system_prices(
        self,
        start_date: str,
        end_date: str,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Collect System Sell Price (SSP) and System Buy Price (SBP).
        Report: B1770
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save: Whether to save data to file
            
        Returns:
            DataFrame with system prices
        """
        logger.info(f"Collecting system prices from {start_date} to {end_date}")
        
        report_type = self.api_config['endpoints']['system_prices']
        
        # Convert dates to required format
        start = parse_date(start_date)
        end = parse_date(end_date)
        
        params = {
            'SettlementDate': start.strftime('%Y-%m-%d'),
            'Period': '*'  # All periods
        }
        
        all_data = []
        
        # Iterate through dates
        current_date = start
        while current_date <= end:
            params['SettlementDate'] = current_date.strftime('%Y-%m-%d')
            
            try:
                response = self._make_request(report_type, params)
                
                # Parse response
                if 'data' in response and 'item' in response['data']:
                    items = response['data']['item']
                    if isinstance(items, dict):
                        items = [items]
                    all_data.extend(items)
                
            except Exception as e:
                logger.error(f"Failed to collect system prices for {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        if not df.empty:
            # Process columns
            if 'settlementDate' in df.columns:
                df['settlement_date'] = pd.to_datetime(df['settlementDate'])
            if 'settlementPeriod' in df.columns:
                df['settlement_period'] = df['settlementPeriod'].astype(int)
            if 'systemSellPrice' in df.columns:
                df['system_sell_price'] = pd.to_numeric(df['systemSellPrice'], errors='coerce')
            if 'systemBuyPrice' in df.columns:
                df['system_buy_price'] = pd.to_numeric(df['systemBuyPrice'], errors='coerce')
            
            logger.info(f"Collected {len(df)} system price records")
            
            if save:
                filename = f"system_prices_{start_date}_{end_date}"
                save_dataframe(df, filename, data_type='raw', format='csv')
        
        return df
    
    def collect_imbalance_prices(
        self,
        start_date: str,
        end_date: str,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Collect imbalance prices (Market Index Price, Market Index Volume).
        Report: B1780
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save: Whether to save data to file
            
        Returns:
            DataFrame with imbalance prices
        """
        logger.info(f"Collecting imbalance prices from {start_date} to {end_date}")
        
        report_type = self.api_config['endpoints']['imbalance_prices']
        
        start = parse_date(start_date)
        end = parse_date(end_date)
        
        params = {
            'SettlementDate': start.strftime('%Y-%m-%d'),
            'Period': '*'
        }
        
        all_data = []
        
        current_date = start
        while current_date <= end:
            params['SettlementDate'] = current_date.strftime('%Y-%m-%d')
            
            try:
                response = self._make_request(report_type, params)
                
                if 'data' in response and 'item' in response['data']:
                    items = response['data']['item']
                    if isinstance(items, dict):
                        items = [items]
                    all_data.extend(items)
                
            except Exception as e:
                logger.error(f"Failed to collect imbalance prices for {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        df = pd.DataFrame(all_data)
        
        if not df.empty:
            if 'settlementDate' in df.columns:
                df['settlement_date'] = pd.to_datetime(df['settlementDate'])
            if 'settlementPeriod' in df.columns:
                df['settlement_period'] = df['settlementPeriod'].astype(int)
            if 'imbalancePrice' in df.columns:
                df['imbalance_price'] = pd.to_numeric(df['imbalancePrice'], errors='coerce')
            
            logger.info(f"Collected {len(df)} imbalance price records")
            
            if save:
                filename = f"imbalance_prices_{start_date}_{end_date}"
                save_dataframe(df, filename, data_type='raw', format='csv')
        
        return df
    
    def collect_generation_by_fuel(
        self,
        start_date: str,
        end_date: str,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Collect actual generation output by fuel type.
        Report: B1620
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save: Whether to save data to file
            
        Returns:
            DataFrame with generation by fuel type
        """
        logger.info(f"Collecting generation by fuel from {start_date} to {end_date}")
        
        report_type = self.api_config['endpoints']['generation_by_fuel']
        
        start = parse_date(start_date)
        end = parse_date(end_date)
        
        params = {
            'SettlementDate': start.strftime('%Y-%m-%d'),
            'Period': '*'
        }
        
        all_data = []
        
        current_date = start
        while current_date <= end:
            params['SettlementDate'] = current_date.strftime('%Y-%m-%d')
            
            try:
                response = self._make_request(report_type, params)
                
                if 'data' in response and 'item' in response['data']:
                    items = response['data']['item']
                    if isinstance(items, dict):
                        items = [items]
                    all_data.extend(items)
                
            except Exception as e:
                logger.error(f"Failed to collect generation data for {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        df = pd.DataFrame(all_data)
        
        if not df.empty:
            if 'settlementDate' in df.columns:
                df['settlement_date'] = pd.to_datetime(df['settlementDate'])
            if 'settlementPeriod' in df.columns:
                df['settlement_period'] = df['settlementPeriod'].astype(int)
            
            logger.info(f"Collected {len(df)} generation records")
            
            if save:
                filename = f"generation_by_fuel_{start_date}_{end_date}"
                save_dataframe(df, filename, data_type='raw', format='csv')
        
        return df
    
    def collect_day_ahead_prices(
        self,
        start_date: str,
        end_date: str,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Collect day-ahead market prices.
        Report: B1430
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save: Whether to save data to file
            
        Returns:
            DataFrame with day-ahead prices
        """
        logger.info(f"Collecting day-ahead prices from {start_date} to {end_date}")
        
        report_type = self.api_config['endpoints']['day_ahead_prices']
        
        start = parse_date(start_date)
        end = parse_date(end_date)
        
        params = {
            'SettlementDate': start.strftime('%Y-%m-%d'),
            'Period': '*'
        }
        
        all_data = []
        
        current_date = start
        while current_date <= end:
            params['SettlementDate'] = current_date.strftime('%Y-%m-%d')
            
            try:
                response = self._make_request(report_type, params)
                
                if 'data' in response and 'item' in response['data']:
                    items = response['data']['item']
                    if isinstance(items, dict):
                        items = [items]
                    all_data.extend(items)
                
            except Exception as e:
                logger.error(f"Failed to collect day-ahead prices for {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        df = pd.DataFrame(all_data)
        
        if not df.empty:
            if 'settlementDate' in df.columns:
                df['settlement_date'] = pd.to_datetime(df['settlementDate'])
            if 'settlementPeriod' in df.columns:
                df['settlement_period'] = df['settlementPeriod'].astype(int)
            
            logger.info(f"Collected {len(df)} day-ahead price records")
            
            if save:
                filename = f"day_ahead_prices_{start_date}_{end_date}"
                save_dataframe(df, filename, data_type='raw', format='csv')
        
        return df
    
    def collect_all_markets(
        self,
        start_date: str,
        end_date: str,
        save: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect data from all available BMRS reports.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save: Whether to save data to files
            
        Returns:
            Dictionary mapping report names to DataFrames
        """
        logger.info(f"Collecting all BMRS data from {start_date} to {end_date}")
        
        data = {}
        
        data['system_prices'] = self.collect_system_prices(
            start_date, end_date, save
        )
        
        data['imbalance_prices'] = self.collect_imbalance_prices(
            start_date, end_date, save
        )
        
        data['generation_by_fuel'] = self.collect_generation_by_fuel(
            start_date, end_date, save
        )
        
        data['day_ahead_prices'] = self.collect_day_ahead_prices(
            start_date, end_date, save
        )
        
        logger.info("Completed collecting all BMRS data")
        
        return data


def main():
    """Example usage of the Elexon BMRS collector."""
    config = load_config()
    
    # Initialize collector
    collector = ElexonBMRSCollector(config)
    
    # Collect data for a sample period
    start_date = "2024-01-01"
    end_date = "2024-01-07"
    
    # Collect all markets
    data = collector.collect_all_markets(start_date, end_date, save=True)
    
    # Print summary
    for market, df in data.items():
        print(f"\n{market.upper()}")
        print(f"Records: {len(df)}")
        if not df.empty:
            print(df.head())


if __name__ == "__main__":
    main()
