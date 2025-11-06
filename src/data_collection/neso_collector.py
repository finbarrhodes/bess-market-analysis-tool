"""
National Grid ESO Data Collection Module

Collects data from National Grid ESO Data Portal including:
- Dynamic Containment (DC)
- Dynamic Regulation (DR)
- Dynamic Moderation (DM)
- System Frequency
- Demand Data
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path
import time
from loguru import logger

from ..utils import (
    load_config,
    setup_logging,
    save_dataframe,
    parse_date,
    generate_date_range
)


class NationalGridESOCollector:
    """Collector for National Grid ESO data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the ESO data collector.
        
        Args:
            config: Configuration dictionary. If None, loads from config.yaml
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self.api_config = config['apis']['national_grid_eso']
        self.base_url = self.api_config['base_url']
        self.rate_limit = self.api_config['rate_limit']
        self.last_request_time = 0
        
        setup_logging(config)
        logger.info("National Grid ESO Collector initialized")
    
    def _rate_limit_wait(self):
        """Implement rate limiting between requests."""
        min_interval = 60 / self.rate_limit  # seconds between requests
        elapsed = time.time() - self.last_request_time
        
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Make HTTP request to ESO API with rate limiting.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response as dictionary
        """
        self._rate_limit_wait()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def collect_dynamic_containment(
        self,
        start_date: str,
        end_date: str,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Collect Dynamic Containment auction results and prices.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save: Whether to save data to file
            
        Returns:
            DataFrame with DC auction data
        """
        logger.info(f"Collecting Dynamic Containment data from {start_date} to {end_date}")
        
        # Note: This is a placeholder structure
        # Actual API endpoints and data structure will need to be verified
        # from the National Grid ESO Data Portal
        
        endpoint = "dynamic-containment"
        
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'format': 'json'
        }
        
        try:
            data = self._make_request(endpoint, params)
            
            # Parse response into DataFrame
            # Structure will depend on actual API response
            if 'records' in data:
                df = pd.DataFrame(data['records'])
            else:
                df = pd.DataFrame(data)
            
            # Process timestamps
            if 'delivery_start' in df.columns:
                df['delivery_start'] = pd.to_datetime(df['delivery_start'])
            if 'delivery_end' in df.columns:
                df['delivery_end'] = pd.to_datetime(df['delivery_end'])
            
            logger.info(f"Collected {len(df)} Dynamic Containment records")
            
            if save:
                filename = f"dc_data_{start_date}_{end_date}"
                save_dataframe(df, filename, data_type='raw', format='csv')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to collect Dynamic Containment data: {e}")
            return pd.DataFrame()
    
    def collect_dynamic_regulation(
        self,
        start_date: str,
        end_date: str,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Collect Dynamic Regulation auction results and prices.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save: Whether to save data to file
            
        Returns:
            DataFrame with DR auction data
        """
        logger.info(f"Collecting Dynamic Regulation data from {start_date} to {end_date}")
        
        endpoint = "dynamic-regulation"
        
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'format': 'json'
        }
        
        try:
            data = self._make_request(endpoint, params)
            
            if 'records' in data:
                df = pd.DataFrame(data['records'])
            else:
                df = pd.DataFrame(data)
            
            if 'delivery_start' in df.columns:
                df['delivery_start'] = pd.to_datetime(df['delivery_start'])
            if 'delivery_end' in df.columns:
                df['delivery_end'] = pd.to_datetime(df['delivery_end'])
            
            logger.info(f"Collected {len(df)} Dynamic Regulation records")
            
            if save:
                filename = f"dr_data_{start_date}_{end_date}"
                save_dataframe(df, filename, data_type='raw', format='csv')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to collect Dynamic Regulation data: {e}")
            return pd.DataFrame()
    
    def collect_system_frequency(
        self,
        start_date: str,
        end_date: str,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Collect system frequency data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save: Whether to save data to file
            
        Returns:
            DataFrame with frequency data
        """
        logger.info(f"Collecting system frequency data from {start_date} to {end_date}")
        
        endpoint = "system-frequency"
        
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'format': 'json'
        }
        
        try:
            data = self._make_request(endpoint, params)
            
            if 'records' in data:
                df = pd.DataFrame(data['records'])
            else:
                df = pd.DataFrame(data)
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"Collected {len(df)} frequency records")
            
            if save:
                filename = f"frequency_data_{start_date}_{end_date}"
                save_dataframe(df, filename, data_type='raw', format='csv')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to collect frequency data: {e}")
            return pd.DataFrame()
    
    def collect_demand_data(
        self,
        start_date: str,
        end_date: str,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Collect historic demand data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save: Whether to save data to file
            
        Returns:
            DataFrame with demand data
        """
        logger.info(f"Collecting demand data from {start_date} to {end_date}")
        
        endpoint = "demand"
        
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'format': 'json'
        }
        
        try:
            data = self._make_request(endpoint, params)
            
            if 'records' in data:
                df = pd.DataFrame(data['records'])
            else:
                df = pd.DataFrame(data)
            
            if 'settlement_date' in df.columns:
                df['settlement_date'] = pd.to_datetime(df['settlement_date'])
            
            logger.info(f"Collected {len(df)} demand records")
            
            if save:
                filename = f"demand_data_{start_date}_{end_date}"
                save_dataframe(df, filename, data_type='raw', format='csv')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to collect demand data: {e}")
            return pd.DataFrame()
    
    def collect_all_markets(
        self,
        start_date: str,
        end_date: str,
        save: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect data from all available markets.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save: Whether to save data to files
            
        Returns:
            Dictionary mapping market names to DataFrames
        """
        logger.info(f"Collecting all market data from {start_date} to {end_date}")
        
        data = {}
        
        # Collect each market type
        data['dynamic_containment'] = self.collect_dynamic_containment(
            start_date, end_date, save
        )
        
        data['dynamic_regulation'] = self.collect_dynamic_regulation(
            start_date, end_date, save
        )
        
        data['system_frequency'] = self.collect_system_frequency(
            start_date, end_date, save
        )
        
        data['demand'] = self.collect_demand_data(
            start_date, end_date, save
        )
        
        logger.info("Completed collecting all market data from National Grid ESO")
        
        return data


def main():
    """Example usage of the National Grid ESO collector."""
    # Load configuration
    config = load_config()
    data_config = config['data_collection']
    
    # Initialize collector
    collector = NationalGridESOCollector(config)
    
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
