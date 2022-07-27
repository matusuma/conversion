SELECT
  ga.user_id AS biano_ident,
  event_name,
  TIMESTAMP_MICROS(ga.event_timestamp) AS timestamp
FROM
  `biano-1152.analytics_{country_id}.events_{date}`  ga
WHERE ga.event_name IN ('session_start', 'purchase') AND ga.user_id IS NOT NULL
