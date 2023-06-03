import ephem
import datetime


def lunar_cycle_from_date(dateobj):
    """Returns float from 0-1. where 0=full, 0.5=new, 1=full"""
    ephem_date = ephem.Date(dateobj)
    nfm = ephem.next_full_moon(dateobj)
    pfm = ephem.previous_full_moon(dateobj)

    lunation = (ephem_date-pfm)/(nfm-pfm)
    return lunation


# date = datetime.date(2022, 2, 16)
#
# print(lunar_cycle_from_date(date))
