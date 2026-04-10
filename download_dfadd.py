from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="isjwdu/DFADD",
    repo_type="dataset",
    local_dir="/nfs/turbo/umd-hafiz/issf_server_data/DFADD",
    local_dir_use_symlinks=False
)