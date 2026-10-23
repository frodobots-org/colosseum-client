"""Earth Rover Mini+ transports: the FrodoBots platform (Agora video, RTM commands and telemetry) or a simulator."""
from __future__ import annotations

from .link import RoverLink, SimulatedRoverLink

__all__ = ['RoverLink', 'SimulatedRoverLink', 'make_link', 'release_link', 'use_shared_link']

_SHARED = {'link': None, 'claim': None, 'release': None}


def use_shared_link(link, *, claim=None, release=None):
    """Register a connection opened ahead of the trial (the operator page's preview); ``None`` clears it.

    ``claim`` runs when a trial adapter takes the link and ``release`` when that adapter
    closes, so the page can hand manual control over and take it back.
    """
    if link is not None:
        link.shared = True
    _SHARED.update(link=link, claim=claim if link is not None else None, release=release if link is not None else None)


def release_link(link):
    """A trial adapter is done with ``link``; a shared link goes back to its owner."""
    if link is _SHARED['link'] and _SHARED['release'] is not None:
        _SHARED['release']()


def make_link(settings, *, cameras, size):
    """Select the transport named by ``adapter_config.link`` without importing hardware SDKs for the simulator."""
    if _SHARED['link'] is not None:
        if _SHARED['claim'] is not None:
            _SHARED['claim']()
        return _SHARED['link']
    kind = settings.get('link', 'frodobots')
    if kind == 'simulated':
        return SimulatedRoverLink(cameras, size)
    if kind == 'frodobots':
        from .frodobots_link import FrodobotsRoverLink
        return FrodobotsRoverLink(settings, cameras, size)
    raise ValueError("adapter_config.link must be 'frodobots' or 'simulated'")
