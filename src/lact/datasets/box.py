"""Adapter for publicly shared Box folders (no login) with resumable downloads.

- Listing: the page ``https://<host>/s/<share>/folder/<id>`` embeds the item list in ``Box.postStreamData``.
  Folders with a vanity name must be listed through ``https://<host>/v/<name>/folder/<id>``; downloads still use
  <share>.
- Download: ``/index.php?rm=box_download_shared_file&shared_name=<share>&file_id=f_<id>`` redirects to a Box download
  server that supports HTTP Range, so interrupted downloads resume where they stopped.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .http_range import USER_AGENT, download_resumable

log = logging.getLogger("lact")
_STREAM_RE = re.compile(r"Box\.postStreamData\s*=\s*(\{.*?\});\s*</script>", re.S)


@dataclass
class BoxItem:
    id: int
    type: str          # file | folder
    name: str
    size: int


class BoxShare:
    def __init__(self, share: str, host: str = "aapm.app.box.com", timeout: float = 60.0,
                 vanity: str | None = None):
        self.share, self.host, self.timeout, self.vanity = share, host, timeout, vanity

    # ------------------------------------------------------------------ listing
    def _page(self, folder_id: int | None, page: int) -> dict:
        base = f"v/{self.vanity}" if self.vanity else f"s/{self.share}"
        url = f"https://{self.host}/{base}" + (f"/folder/{folder_id}" if folder_id else "")
        if page > 1:
            url += f"?page={page}"
        html = urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=self.timeout).read()
        m = _STREAM_RE.search(html.decode("utf-8", "replace"))
        if not m:
            raise RuntimeError(f"Cannot read the Box folder listing from {url} (has the Box page changed?)")
        return json.loads(m.group(1))["/app-api/enduserapp/shared-folder"]

    def list_folder(self, folder_id: int | None = None) -> list[BoxItem]:
        first = self._page(folder_id, 1)
        raw = list(first["items"])
        for p in range(2, int(first.get("pageCount", 1)) + 1):
            raw += self._page(folder_id, p)["items"]
        return [BoxItem(int(i["id"]), i["type"], i["name"], int(i.get("itemSize") or 0)) for i in raw]

    def find(self, path: str) -> BoxItem:
        """Find an item by its path in the shared folder.

        Example: 'Patient_Data/Training_Image_Data/3mm B30/full_3mm.zip'.
        """
        folder: int | None = None
        item: BoxItem | None = None
        for part in [p for p in path.split("/") if p]:
            items = {i.name: i for i in self.list_folder(folder)}
            if part not in items:
                raise FileNotFoundError(f"'{part}' not found in Box (available: {sorted(items)})")
            item = items[part]
            folder = item.id
        if item is None:
            raise ValueError("Empty path")
        return item

    # ------------------------------------------------------------------ download
    def download_url(self, file_id: int) -> str:
        return (f"https://{self.host}/index.php?rm=box_download_shared_file&shared_name={self.share}"
                f"&file_id=f_{file_id}")

    def download(self, item: BoxItem, dst: str | os.PathLike, retries: int = 8, chunk: int = 8 << 20) -> Path:
        """Download to ``dst`` (through ``dst.partial``, resuming an interrupted download). Checks the final size."""
        return download_resumable(self.download_url(item.id), dst, item.size, headers=USER_AGENT, retries=retries,
                                  chunk=chunk, timeout=self.timeout)
