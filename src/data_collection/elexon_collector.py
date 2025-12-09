"""
Elexon BMRS Data Collection Module

Collects data from Elexon Balancing Mechanism Reporting Service including:
- System Buy/Sell Prices
- Imbalance Prices
- Balancing Mechanism actions
- Generation by fuel type
- Day-ahead prices
"""

import pandas as pd
from typing import Optional, Dict
from loguru import logger
from ElexonDataPortal import api


from ..utils import (
    load_config,
    setup_logging,
    save_dataframe,
    get_api_key
)


class ElexonBMRSCollector:
    """Collector for Elexon BMRS data using ElexonDataPortal package."""

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

        # Get API key from environment
        try:
            api_key = get_api_key('ELEXON')
        except ValueError as e:
            logger.error(f"API key not found: {e}")
            raise ValueError("API key required for Elexon BMRS. Set ELEXON_API_KEY in .env file")

        # Initialize ElexonDataPortal client
        self.client = api.Client(api_key)

        setup_logging(config)
        logger.info("Elexon BMRS Collector initialized with ElexonDataPortal")
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and types across all data.

        Args:
            df: Raw DataFrame from API

        Returns:
            Processed DataFrame
        """
        if df.empty:
            return df

        # Standardize datetime columns
        if 'settlementDate' in df.columns:
            df['settlement_date'] = pd.to_datetime(df['settlementDate'])
        if 'settlementPeriod' in df.columns:
            df['settlement_period'] = pd.to_numeric(df['settlementPeriod'], errors='coerce').astype('Int64')

        # Standardize price columns
        price_columns = {
            'systemSellPrice': 'system_sell_price',
            'systemBuyPrice': 'system_buy_price',
            'imbalancePrice': 'imbalance_price',
            'imbalancePriceAmount': 'imbalance_price'
        }

        for old_col, new_col in price_columns.items():
            if old_col in df.columns:
                df[new_col] = pd.to_numeric(df[old_col], errors='coerce')

        return df

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

        try:
            # Use ElexonDataPortal to get system prices
            # API uses start_date and end_date parameters
            df = self.client.get_B1770(
                start_date=start_date,
                end_date=end_date
            )

            # Process DataFrame
            df = self._process_dataframe(df)

            logger.info(f"Collected {len(df)} system price records")

            if save and not df.empty:
                filename = f"system_prices_{start_date}_{end_date}"
                save_dataframe(df, filename, data_type='raw', format='csv')

        except Exception as e:
            logger.error(f"Failed to collect system prices: {e}")
            df = pd.DataFrame()

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

        try:
            df = self.client.get_B1780(
                start_date=start_date,
                end_date=end_date
            )

            df = self._process_dataframe(df)

            logger.info(f"Collected {len(df)} imbalance price records")

            if save and not df.empty:
                filename = f"imbalance_prices_{start_date}_{end_date}"
                save_dataframe(df, filename, data_type='raw', format='csv')

        except Exception as e:
            logger.error(f"Failed to collect imbalance prices: {e}")
            df = pd.DataFrame()

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

        try:
            df = self.client.get_B1620(
                start_date=start_date,
                end_date=end_date
            )

            df = self._process_dataframe(df)

            logger.info(f"Collected {len(df)} generation records")

            if save and not df.empty:
                filename = f"generation_by_fuel_{start_date}_{end_date}"
                save_dataframe(df, filename, data_type='raw', format='csv')

        except Exception as e:
            logger.error(f"Failed to collect generation data: {e}")
            df = pd.DataFrame()

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

        try:
            df = self.client.get_B1430(
                start_date=start_date,
                end_date=end_date
            )

            df = self._process_dataframe(df)

            logger.info(f"Collected {len(df)} day-ahead price records")

            if save and not df.empty:
                filename = f"day_ahead_prices_{start_date}_{end_date}"
                save_dataframe(df, filename, data_type='raw', format='csv')

        except Exception as e:
            logger.error(f"Failed to collect day-ahead prices: {e}")
            df = pd.DataFrame()

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
