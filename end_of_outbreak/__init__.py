"""End-of-outbreak decision-making under superspreading and onset-date delays.

Deliberately thin: modules are imported directly (``from end_of_outbreak import
delay_distributions``) rather than re-exported here. The Snakemake workflow lists the exact
package modules each rule depends on, and a fat ``__init__`` that every rule had to depend on
would defeat that scheme.
"""

__version__ = "0.1.0"
