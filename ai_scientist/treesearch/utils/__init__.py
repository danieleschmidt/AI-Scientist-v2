import logging
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger("ai-scientist")


def copytree(src: Path, dst: Path, use_symlinks=True):
    """
    Copy contents of `src` to `dst`. Unlike shutil.copytree, the dst dir can exist and will be merged.
    If src is a file, only that file will be copied. Optionally uses symlinks instead of copying.

    Args:
        src (Path): source directory
        dst (Path): destination directory
    """
    assert dst.is_dir()

    if src.is_file():
        dest_f = dst / src.name
        assert not dest_f.exists(), dest_f
        if use_symlinks:
            (dest_f).symlink_to(src)
        else:
            shutil.copyfile(src, dest_f)
        return

    for f in src.iterdir():
        dest_f = dst / f.name
        assert not dest_f.exists(), dest_f
        if use_symlinks:
            (dest_f).symlink_to(f)
        elif f.is_dir():
            shutil.copytree(f, dest_f)
        else:
            shutil.copyfile(f, dest_f)


def clean_up_dataset(path: Path):
    for item in path.rglob("__MACOSX"):
        if item.is_dir():
            shutil.rmtree(item)
    for item in path.rglob(".DS_Store"):
        if item.is_file():
            item.unlink()


def extract_archives(path: Path, max_depth: int = 3):
    """
    Recursively unzips all .zip files within `path` and cleans up task dir
    
    Args:
        path: Directory to search for zip files
        max_depth: Maximum recursion depth for nested zips (prevents infinite loops)
    """
    if max_depth <= 0:
        logger.warning(f"Maximum extraction depth reached, stopping recursion")
        return
    
    extracted_any = False
    
    for zip_f in path.rglob("*.zip"):
        f_out_dir = zip_f.with_suffix("")

        # special case: the intended output path already exists (maybe data has already been extracted by user)
        if f_out_dir.exists():
            logger.debug(
                f"Skipping {zip_f} as an item with the same name already exists."
            )
            # if it's a file, it's probably exactly the same as in the zip -> remove the zip
            # Enhanced validation: check if file size matches (basic content validation)
            if f_out_dir.is_file() and f_out_dir.suffix != "":
                try:
                    # Get original file size from zip
                    with zipfile.ZipFile(zip_f, "r") as zip_ref:
                        zip_info = zip_ref.infolist()
                        if len(zip_info) == 1 and zip_info[0].file_size == f_out_dir.stat().st_size:
                            logger.debug(f"File size matches, removing zip: {zip_f}")
                            zip_f.unlink()
                        else:
                            logger.debug(f"File size mismatch, keeping zip: {zip_f}")
                except (zipfile.BadZipFile, OSError) as e:
                    logger.warning(f"Could not validate zip content: {e}")
            continue

        logger.debug(f"Extracting: {zip_f}")
        f_out_dir.mkdir(exist_ok=True)
        
        try:
            # Import security validation
            from ai_scientist.utils.input_validation import safe_extract_zip, SecurityError
            
            # Secure extraction
            try:
                safe_extract_zip(str(zip_f), str(f_out_dir))
                extracted_any = True
            except SecurityError as se:
                logger.error(f"Security validation failed for {zip_f}: {se}")
                f_out_dir.rmdir()  # Clean up empty directory
                continue
        except zipfile.BadZipFile as e:
            logger.error(f"Failed to extract {zip_f}: {e}")
            f_out_dir.rmdir()  # Clean up empty directory
            continue

        # remove any unwanted files
        clean_up_dataset(f_out_dir)

        contents = list(f_out_dir.iterdir())

        # special case: the zip contains a single dir/file with the same name as the zip
        if len(contents) == 1 and contents[0].name == f_out_dir.name:
            sub_item = contents[0]
            # if it's a dir, move its contents to the parent and remove it
            if sub_item.is_dir():
                logger.debug(f"Special handling (child is dir) enabled for: {zip_f}")
                for f in sub_item.rglob("*"):
                    if f.is_file():  # Only move files, not directories
                        try:
                            # Securely calculate relative path
                            from ai_scientist.utils.path_security import secure_relative_path, validate_safe_path
                            
                            relative_path = secure_relative_path(f, sub_item)
                            target_path = f_out_dir / relative_path
                            
                            # Validate that target path is safe
                            if not validate_safe_path(target_path, f_out_dir, allow_relative=False):
                                logger.warning(f"Skipping potentially unsafe file move: {f} -> {target_path}")
                                continue
                            
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(f), str(target_path))
                            
                        except Exception as e:
                            logger.warning(f"Failed to move file {f} safely: {e}")
                            continue
                shutil.rmtree(sub_item)  # Use rmtree for recursive removal
            # if it's a file, rename it to the parent and remove the parent
            elif sub_item.is_file():
                logger.debug(f"Special handling (child is file) enabled for: {zip_f}")
                sub_item_tmp = sub_item.rename(f_out_dir.with_suffix(".__tmp_rename"))
                f_out_dir.rmdir()
                sub_item_tmp.rename(f_out_dir)

        zip_f.unlink()

    # Handle nested zips: if we extracted any files, check for new zip files
    if extracted_any and max_depth > 1:
        logger.debug(f"Checking for nested zip files (depth: {max_depth})")
        extract_archives(path, max_depth - 1)


def preproc_data(path: Path):
    extract_archives(path)
    clean_up_dataset(path)
