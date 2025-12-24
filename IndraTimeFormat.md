# IndraTime Format Description

This document describes the string representation of time used by `IndraTime` in `IndraLib`. This format is designed to represent deep time (geological scales) as well as historical and modern dates.

## Time Points

A time point is represented as a string. The parsing logic is case-insensitive.

### Standard Date (AD/CE)
Standard historical dates are represented in ISO-like format.
- Format: `YYYY-MM-DD`, `YYYY-MM`, or `YYYY`.
- Examples: `2020-01-01`, `1999-12`, `2025`.
- Suffix: An optional `AD` suffix is allowed (e.g., `2020 AD`), but usually omitted.

### Deep Time and Historical Representations
For dates before 1 AD, various scales are used based on the magnitude of time.

#### BC (Before Christ)
Used for historical dates before 1 AD.
- Format: `Y BC`, `Y-M BC`, or `Y-M-D BC`.
- Logic: `1 BC` corresponds to astronomical year 0. `2 BC` is year -1.
- Example: `2020 BC`, `44-03-15 BC`.

#### BP (Before Present)
Used for radiocarbon dating and prehistory. "Present" is defined as 1950 AD.
- Format: `N BP`
- Logic: `Year = 1950 - N`.
- Example: `1000 BP` (corresponds to 950 AD), `5000 BP` (3050 BC).

#### kya / ka (Thousand Years Ago)
Used for longer prehistoric timescales.
- Format: `N kya BP`, `N ka BP`, `N kyr BP`, `N kya`, `N ka`, `N kyr`.
- Logic: `Year = 1950 - (N * 1,000)`.
- Example: `10 kya` (8050 BC).

#### Ma / mya (Million Years Ago)
Used for geological timescales.
- Format: `N Ma BP`, `N Ma`, `N mya`, `N mya BP`.
- Logic: `Year = 1950 - (N * 1,000,000)`.
- Example: `65 Ma` (approx. extinction of dinosaurs).

#### Ga / bya (Billion Years Ago)
Used for cosmological and early Earth timescales.
- Format: `N Ga BP`, `N Ga`, `N bya`, `N bya BP`.
- Logic: `Year = 1950 - (N * 1,000,000,000)`.
- Example: `4.5 Ga` (approx. age of Earth).

## Time Intervals

A time interval represents a span between two time points.

- Format: `Start - End`
- Separator: `" - "` (space, hyphen, space).
- Logic: Parses two discrete time points.
- Examples: 
  - `2020-01-01 - 2021-01-01`
  - `100 BC - 50 BC`
  - `65 Ma - 2.5 Ma`

## Output formatting (Canonical Representation)

When converting from internal Julian Date (float) back to string, `IndraTime` applies the following rules to select the appropriate unit:

### Dates >= 1 AD
- Represented as `YYYY-MM-DD`.
- If Month and Day are 1, and Year < 1900: returns `YYYY`.
- If Day is 1, and Year < 1900: returns `YYYY-MM`.

### Dates < 1 AD
The format depends on the age relative to 1 AD (1721423.5 JD):

1. **Recent BC**: Date > (1 AD - 13,000 years)
   - Format: `Y BC`
   
2. **Prehistory (BP)**: Date > (1 AD - 100,000 years)
   - Format: `N BP`
   - Calculated from 1950.
   
3. **Paleo (kya)**: Date > (1 AD - 100,000,000 years)
   - Format: `N kya BP`
   - Rounded to 2 decimal places.
   
4. **Geological (Ma)**: Date > (1 AD - 10,000,000,000 years)
   - Format: `N Ma BP`
   - Rounded to 2 decimal places.
   
5. **Cosmological (Ga)**: Older dates
   - Format: `N Ga BP`
   - Rounded to 3 decimal places.
