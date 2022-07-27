SELECT
  pixel.biano_ident,
  pixel.event_type,
  timestamp
FROM
  `biano-pixel.raw_requests.valid_biano_{country}` pixel
WHERE
  pixel.cookies_enabled = TRUE
  AND
  DATE(timestamp) = '{date}'
