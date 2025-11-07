
class GoogleDriveContextManager:
    """
    Manages caching and formatting of Google Drive content for PR analysis.
    
    Note: Actual searching and fetching is done via the MaaS Google Drive MCP.
    This class just coordinates and caches the results.
    """
    
    def __init__(self):
        self.cache = {}  # Simple in-memory cache: {filename: content}
        self.gdrive_enabled = True
    
    def add_file_to_cache(self, file_name: str, content: str):
        """Add a file's content to the cache."""
        self.cache[file_name] = content
        print(f"✓ Cached: {file_name} ({len(content)} chars)", file=sys.stderr)
    
    def get_cached_file(self, file_name: str) -> str | None:
        """Get a file's content from cache if it exists."""
        return self.cache.get(file_name)
    
    def build_context_from_files(self, file_names: list[str]) -> str:
        """
        Build context prompt section from specified Google Drive files.
        
        This method generates instructions for the user to fetch files via MaaS Google Drive MCP,
        since the actual file fetching must be done through Cursor's MCP interface.
        
        Args:
            file_names: List of file names to include
                       e.g., ["ThunderQ4Plan", "ThunderBestPractices"]
            
        Returns:
            Instructions for fetching files, or cached content if available
        """
        if not self.gdrive_enabled or not file_names:
            return ""
        
        # Check if all files are in cache
        cached_parts = []
        missing_files = []
        
        for file_name in file_names:
            cached_content = self.get_cached_file(file_name)
            if cached_content:
                cached_parts.append(f"### {file_name}\n{cached_content}\n")
                print(f"✓ Using cached: {file_name}", file=sys.stderr)
            else:
                missing_files.append(file_name)
        
        # If files are missing, return instructions
        if missing_files:
            print(f"⚠️  Files need to be fetched via MaaS Google Drive MCP:", file=sys.stderr)
            for fname in missing_files:
                print(f"   - {fname}", file=sys.stderr)
            
            instructions = f"""
## INSTRUCTIONS TO FETCH GOOGLE DRIVE FILES

The following files need to be fetched from Google Drive:
{chr(10).join(f'- {fname}' for fname in missing_files)}

Please use the MaaS Google Drive MCP to fetch them:

1. Search for each file:
   mcp_MaaS_Google_Drive_gdrive_search(query="{missing_files[0]}")

2. Get the file content:
   mcp_MaaS_Google_Drive_gdrive_get_file(file_url="<url_from_search>")

3. Then call this analysis again with the cached content, or provide the URLs directly.

OR provide full Google Drive URLs instead of file names:
   gdrive_files=["https://drive.google.com/file/d/YOUR_FILE_ID/view"]
"""
            return instructions
        
        # All files are cached, build the context
        if cached_parts:
            return "\n".join([
                "## REFERENCE DOCUMENTATION",
                "The following documentation should be used to calibrate this analysis:",
                "",
                *cached_parts
            ])
        
        return ""
    
    def disable(self):
        """Disable Google Drive integration (fallback if it fails)"""
        self.gdrive_enabled = False
        print("Google Drive integration disabled", file=sys.stderr)
