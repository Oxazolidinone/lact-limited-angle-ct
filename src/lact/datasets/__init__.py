"""Dataset adapters: download, read and split the raw data of each dataset.

    box.py            public Box folders (listing, resumable downloads)
    http_range.py     HTTP Range reads and resumable downloads shared by the downloaders
    dicom.py          DICOM CT series (Mayo/AAPM) and synthetic DICOM slices
    mayo.py           Mayo/AAPM 2016 Low Dose CT Grand Challenge download
    deeplesion/       NIH DeepLesion: 16-bit PNG reader, download, patient and key-slice splits
    detect/           2DeteCT: measured sinograms, LION split, partial downloads from Zenodo

Layer: below ``services`` (which turns raw data into the image and sinogram stores) and above ``infra``.
"""
