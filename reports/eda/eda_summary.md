# EDA summary

> Placeholder. Fill in after running `python src/02_eda.py` and reading the generated
> figures in this directory plus `reports/cleaning_summary.md`. This file is not
> auto-generated — it is a one-page judgment call on top of the plots.

For each of the 9 analyst hypotheses (spec section 4), record: did the two-panel plot
show a real, CI-separated gap, or did the groups overlap within their Wilson intervals?

1. Amount — `amount_usd`, `account_balance_usd` (see `hist_amount_usd.png`, `decile_amount_usd.png`)
2. Velocity — `hours_since_last_txn`, `txn_count_last_24h`, `velocity_score`
3. Location & network — `distance_from_home_km`, `hypothesis_is_foreign_transaction.png`, `hypothesis_used_vpn.png`, `pair_is_foreign_transaction_x_used_vpn.png`
4. Authentication — `hypothesis_auth_method.png`, `pair_auth_method_x_channel.png`
5. Device & channel — `hypothesis_device_type.png`, `pair_device_type_x_channel.png`
6. Merchant — `hypothesis_merchant_category.png`, `pair_channel_x_merchant_category.png`, `hypothesis_is_new_merchant.png`
7. Card & customer profile — `card_age_months`, `prior_disputes`, `customer_age`
8. Time — `hour_of_day.png`, `day_of_week.png`, `heatmap_hour_day.png`
9. Scam signals — `hypothesis_is_ai_generated_scam_attempt.png`, `hypothesis_billing_shipping_mismatch.png`

Also record: anything surprising, `missingness.csv` findings, `mutual_information.png`
ranking vs. intuition, and the adversarial-validation AUC from `reports/cleaning_summary.md`.
