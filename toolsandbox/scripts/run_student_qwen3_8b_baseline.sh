#!/usr/bin/env bash
set -euo pipefail

VLLM_URL="http://localhost:8881/v1"
PARALLEL=1
OUTPUT_DIR="data"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vllm-url) VLLM_URL="$2"; shift 2 ;;
    -p|--parallel) PARALLEL="$2"; shift 2 ;;
    -o|--output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "[Qwen3_8B baseline] VLLM_URL=$VLLM_URL"

cd /c2/taeil/ToolSandbox && /c2/taeil/anaconda3/envs/ToolSandbox/bin/tool_sandbox \
  --agent Qwen3_8B \
  --user GPT_5_Mini \
  --vllm-url "$VLLM_URL" \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR" \
  --scenarios \
    add_contact_with_name_and_phone_number \
    add_reminder_content_and_date_and_time \
    add_reminder_content_and_date_and_time_alt \
    add_reminder_content_and_date_and_time_multiple_user_turn \
    add_reminder_content_and_date_and_time_multiple_user_turn_alt \
    add_reminder_content_and_week_delta_and_time \
    add_reminder_content_and_week_delta_and_time_alt \
    add_reminder_content_and_week_delta_and_time_and_location \
    add_reminder_content_and_week_delta_and_time_and_location_alt \
    add_reminder_content_and_week_delta_and_time_and_location_low_battery_mode_multiple_user_turn_alt \
    add_reminder_content_and_week_delta_and_time_and_location_multiple_user_turn \
    add_reminder_content_and_week_delta_and_time_and_location_multiple_user_turn_alt \
    add_reminder_content_and_week_delta_and_time_multiple_user_turn \
    add_reminder_content_and_week_delta_and_time_multiple_user_turn_alt \
    add_reminder_content_and_weekday_delta_and_time \
    add_reminder_content_and_weekday_delta_and_time_alt \
    add_reminder_content_and_weekday_delta_and_time_multiple_user_turn \
    add_reminder_content_and_weekday_delta_and_time_multiple_user_turn_alt \
    cellular_off \
    convert_currency \
    convert_currency_canonicalize \
    find_address_with_lat_lon \
    find_current_city_insufficient_information \
    find_current_city_low_battery_mode \
    find_current_city_low_battery_mode_alt \
    find_current_city_low_battery_mode_insufficient_information \
    find_current_location_insufficient_information \
    find_current_location_low_battery_mode_insufficient_information \
    find_days_till_holiday \
    find_days_till_holiday_alt \
    find_days_till_holiday_insufficient_information \
    find_days_till_holiday_insufficient_information_alt \
    find_days_till_holiday_multiple_user_turn \
    find_days_till_holiday_wifi_off \
    find_days_till_holiday_wifi_off_alt \
    find_days_till_holiday_wifi_off_multiple_user_turn \
    find_distance_with_location_name \
    find_distance_with_location_name_alt \
    find_distance_with_location_name_insufficient_information \
    find_distance_with_location_name_insufficient_information_alt \
    find_distance_with_location_name_low_battery_mode_multiple_user_turn \
    find_distance_with_location_name_multiple_user_turn \
    find_min_temperature_weekday_insufficient_information \
    find_min_temperature_weekday_insufficient_information_alt \
    find_phone_number_with_location_name \
    find_stock_symbol_with_company_name \
    find_stock_symbol_with_company_name_low_battery_mode \
    find_stock_symbol_with_company_name_low_battery_mode_alt \
    find_temperature \
    find_temperature_f_with_location \
    find_temperature_f_with_location_alt \
    find_temperature_f_with_location_and_time_diff_low_battery_mode_multiple_user_turn \
    find_temperature_f_with_location_and_time_diff_multiple_user_turn \
    find_temperature_f_with_location_and_time_diff_wifi_off_multiple_user_turn \
    find_temperature_f_with_location_insufficient_information \
    find_temperature_f_with_location_insufficient_information_alt \
    find_temperature_f_with_location_wifi_off \
    find_temperature_f_with_location_wifi_off_alt \
    find_temperature_low_battery_mode \
    find_temperature_low_battery_mode_alt \
    find_thanksgiving_timestamp \
    get_cellular \
    get_wifi \
    modify_contact_with_message_recency \
    modify_contact_with_message_recency_alt \
    modify_contact_with_message_recency_insufficient_information \
    modify_contact_with_message_recency_insufficient_information_alt \
    modify_contact_with_message_recency_multiple_user_turn \
    modify_contact_with_message_recency_multiple_user_turn_alt \
    modify_reminder_with_recency_latest \
    modify_reminder_with_recency_latest_alt \
    modify_reminder_with_recency_latest_insufficient_information \
    remove_contact_by_phone \
    remove_contact_by_phone_alt \
    remove_contact_by_phone_ambiguous \
    remove_contact_by_phone_ambiguous_alt \
    remove_contact_by_phone_multiple_user_turn \
    remove_contact_by_phone_multiple_user_turn_alt \
    remove_contact_by_phone_no_remove_contact_insufficient_information \
    remove_contact_by_phone_no_remove_contact_insufficient_information_alt \
    remove_contact_by_phone_no_search_contacts_insufficient_information \
    remove_contact_by_phone_no_search_contacts_insufficient_information_alt \
    remove_contact_with_id \
    remove_reminder_with_recency_latest \
    remove_reminder_with_recency_latest_alt \
    remove_reminder_with_recency_latest_insufficient_information \
    search_message_with_recency_latest \
    search_message_with_recency_latest_alt \
    search_message_with_recency_latest_multiple_user_turn \
    search_message_with_recency_latest_multiple_user_turn_alt \
    search_message_with_recency_oldest \
    search_message_with_recency_oldest_alt \
    search_message_with_recency_oldest_multiple_user_turn \
    search_message_with_recency_oldest_multiple_user_turn_alt \
    search_name_with_relationship \
    search_phone_number_with_name \
    search_relationship_with_phone_number \
    search_reminder_with_creation_recency_yesterday \
    search_reminder_with_creation_recency_yesterday_implicit \
    search_reminder_with_creation_recency_yesterday_insufficient_information \
    search_reminder_with_creation_recency_yesterday_insufficient_information_implicit \
    search_reminder_with_recency_upcoming \
    search_reminder_with_recency_upcoming_implicit \
    search_reminder_with_recency_upcoming_insufficient_information \
    search_reminder_with_recency_upcoming_insufficient_information_implicit \
    search_reminder_with_recency_yesterday \
    search_reminder_with_recency_yesterday_implicit \
    search_reminder_with_recency_yesterday_insufficient_information \
    search_reminder_with_recency_yesterday_insufficient_information_implicit \
    search_sender_phone_number_with_content \
    send_message_with_contact_content_cellular_off \
    send_message_with_contact_content_cellular_off_alt \
    send_message_with_contact_content_cellular_off_insufficient_information \
    send_message_with_contact_content_cellular_off_insufficient_information_alt \
    send_message_with_contact_content_cellular_off_multiple_user_turn \
    send_message_with_contact_content_cellular_off_multiple_user_turn_alt \
    send_message_with_phone_number_and_content \
    turn_on_cellular_low_battery_mode \
    turn_on_cellular_low_battery_mode_implicit \
    turn_on_location_low_battery_mode \
    turn_on_location_low_battery_mode_implicit \
    turn_on_wifi_low_battery_mode \
    turn_on_wifi_low_battery_mode_implicit \
    update_contact_relationship_with_relationship \
    update_contact_relationship_with_relationship_alt \
    update_contact_relationship_with_relationship_multiple_user_turn \
    update_contact_relationship_with_relationship_twice_multiple_user_turn \
    update_contact_with_id_and_phone_number \
    wifi_off
