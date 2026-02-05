#!/usr/bin/env python3
"""
Simple unit test for memory bank DB functions.

Tests the core database functionality without requiring full environment setup.
"""

import asyncio
import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engram_mcp import db as dbmod


async def test_fetch_virtual_memory_files():
    """Test the fetch_virtual_memory_files function."""
    print("🧪 Testing fetch_virtual_memory_files function...")

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test.db")

        # Initialize database
        await dbmod.init_db(db_path)

        # Create a test project
        project_id = "test_project"
        await dbmod.create_project(
            db_path,
            project_id=project_id,
            project_name="Test Project",
            project_type="test",
            directory=None,
            embedding_dim=384,
        )

        print("✓ Test project created")

        # Test 1: Empty memory bank
        print("\n1. Testing empty memory bank...")
        result = await dbmod.fetch_virtual_memory_files(db_path, project_id)
        assert result == {}, f"Expected empty dict, got {result}"
        print("   ✓ Returns empty dict when no memory files exist")

        # Test 2: Add a memory bank file manually
        print("\n2. Adding memory bank file...")
        chunk_id = "test_chunk_id_123"
        content = "Test memory bank content about OAuth authentication"
        vfs_path = "vfs://memory/activeContext.md"

        # Reserve internal ID
        await dbmod.reserve_internal_ids(db_path, project_id, 1)

        # Insert chunk
        await dbmod.upsert_chunks(
            db_path,
            project_id=project_id,
            rows=[(chunk_id, 0, content, 100, {
                "type": "memory_bank",
                "section": "activeContext",
            })],
        )

        # Insert file metadata
        from engram_mcp.db import FileMetadataRow
        await dbmod.upsert_file_metadata(
            db_path,
            project_id=project_id,
            rows=[
                FileMetadataRow(
                    file_path=vfs_path,
                    mtime_ns=0,
                    size_bytes=len(content),
                    content_hash="test_hash",
                    chunk_ids=[],
                )
            ],
        )

        # Insert file_chunks mapping
        async with dbmod.get_connection(db_path) as db:
            await db.execute(
                "INSERT INTO file_chunks (project_id, file_path, chunk_id) VALUES (?, ?, ?)",
                (project_id, vfs_path, chunk_id),
            )
            await db.commit()

        print(f"   ✓ Added chunk to {vfs_path}")

        # Test 3: Fetch the memory bank file
        print("\n3. Fetching memory bank files...")
        result = await dbmod.fetch_virtual_memory_files(db_path, project_id)
        print(f"   Result: {result}")
        assert "activeContext" in result, "activeContext not in result!"
        assert result["activeContext"] == content, f"Content mismatch! Expected '{content}', got '{result['activeContext']}'"
        print("   ✓ Successfully fetched memory bank content")

        # Test 4: Add multiple sections
        print("\n4. Adding multiple memory sections...")
        sections = [
            ("productContext", "Building authentication system"),
            ("techContext", "Using FastAPI and PostgreSQL"),
        ]

        for idx, (section, section_content) in enumerate(sections, start=1):
            section_chunk_id = f"test_chunk_{section}"
            section_vfs_path = f"vfs://memory/{section}.md"

            await dbmod.upsert_chunks(
                db_path,
                project_id=project_id,
                rows=[(section_chunk_id, idx, section_content, 50, {
                    "type": "memory_bank",
                    "section": section,
                })],
            )

            await dbmod.upsert_file_metadata(
                db_path,
                project_id=project_id,
                rows=[
                    FileMetadataRow(
                        file_path=section_vfs_path,
                        mtime_ns=0,
                        size_bytes=len(section_content),
                        content_hash=f"hash_{section}",
                        chunk_ids=[],
                    )
                ],
            )

            async with dbmod.get_connection(db_path) as db:
                await db.execute(
                    "INSERT INTO file_chunks (project_id, file_path, chunk_id) VALUES (?, ?, ?)",
                    (project_id, section_vfs_path, section_chunk_id),
                )
                await db.commit()

        print("   ✓ Added multiple sections")

        # Test 5: Fetch all sections
        print("\n5. Fetching all memory sections...")
        result = await dbmod.fetch_virtual_memory_files(db_path, project_id)
        print(f"   Found {len(result)} sections: {list(result.keys())}")
        assert len(result) == 3, f"Expected 3 sections, got {len(result)}"
        assert "activeContext" in result
        assert "productContext" in result
        assert "techContext" in result
        print("   ✓ All sections retrieved correctly")

        # Test 6: Verify content preservation
        print("\n6. Verifying content preservation...")
        assert result["productContext"] == "Building authentication system"
        assert result["techContext"] == "Using FastAPI and PostgreSQL"
        print("   ✓ Content preserved correctly")

        # Test 7: Test prefix filtering
        print("\n7. Testing prefix filtering...")
        # This should return the same results since all our files use the default prefix
        result_with_prefix = await dbmod.fetch_virtual_memory_files(
            db_path, project_id, prefix="vfs://memory/"
        )
        assert len(result_with_prefix) == 3
        print("   ✓ Prefix filtering works")

        # Add a non-memory virtual file to test filtering
        other_vfs_path = "vfs://insights/test.md"
        other_chunk_id = "other_chunk_123"
        await dbmod.upsert_chunks(
            db_path,
            project_id=project_id,
            rows=[(other_chunk_id, 10, "other content", 20, {"type": "insight"})],
        )
        await dbmod.upsert_file_metadata(
            db_path,
            project_id=project_id,
            rows=[
                FileMetadataRow(
                    file_path=other_vfs_path,
                    mtime_ns=0,
                    size_bytes=13,
                    content_hash="other_hash",
                    chunk_ids=[],
                )
            ],
        )
        async with dbmod.get_connection(db_path) as db:
            await db.execute(
                "INSERT INTO file_chunks (project_id, file_path, chunk_id) VALUES (?, ?, ?)",
                (project_id, other_vfs_path, other_chunk_id),
            )
            await db.commit()

        # Fetch memory files again - should still be 3
        memory_only = await dbmod.fetch_virtual_memory_files(db_path, project_id)
        assert len(memory_only) == 3, f"Expected 3 memory files, got {len(memory_only)}"
        print("   ✓ Only memory bank files returned (other virtual files excluded)")

        print("\n✅ All tests passed!")
        return True


if __name__ == "__main__":
    try:
        result = asyncio.run(test_fetch_virtual_memory_files())
        if result:
            print("\n🎉 Memory Bank DB functions are working correctly!")
            exit(0)
        else:
            print("\n❌ Some tests failed!")
            exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
