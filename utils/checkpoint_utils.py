from pathlib import Path


def search_checkpoint(run_path, mode='best'):
    def search_best_checkpoint(checkpoint_fld):
        checkpoint_path = None
        _best_found = False
        for checkpoint in checkpoint_fld.glob('*'):
            with open(checkpoint / 'aliases.txt', 'r') as f:
                aliases = f.readline()
            if 'best' in aliases:
                checkpoint_path = next(iter(checkpoint.glob('*.ckpt')))
                _best_found = checkpoint_path.is_file()
        return checkpoint_path, _best_found

    def search_last_checkpoint(checkpoint_fld):
        checkpoint_path = None
        _last_found = False
        for checkpoint in checkpoint_fld.glob('*'):
            if checkpoint.name == 'last':
                checkpoint_path = next(iter(checkpoint.glob('*.ckpt')))
                _last_found = checkpoint_path.is_file()
        return checkpoint_path, _last_found

    run_path = Path(run_path)
    checkpoint_fld = next(iter(run_path.rglob('checkpoints')))
    _checkpoint_found = False
    if mode == 'best':
        checkpoint_path, _checkpoint_found = search_best_checkpoint(checkpoint_fld)
    elif mode == 'last':
        checkpoint_path, _checkpoint_found = search_last_checkpoint(checkpoint_fld)
    elif mode == 'best-preferred':
        checkpoint_path, _checkpoint_found = search_best_checkpoint(checkpoint_fld)
        if not _checkpoint_found:
            checkpoint_path, _checkpoint_found = search_last_checkpoint(checkpoint_fld)
    else:
        raise NotImplementedError('mode must in {"best", "last", "best-preferred"}')

    if not _checkpoint_found:
        raise FileNotFoundError('Cannot find model checkpoint')
    return checkpoint_path