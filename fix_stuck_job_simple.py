"""
Simple script to fix stuck job without loading AI models
"""
import uuid
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

JOB_ID = "88e3a6cc-37c3-4ff0-b023-4051c347cb72"

# Create engine directly - use default if not in .env
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    r"mssql+pyodbc://transapp:transapp@(localdb)\MSSQLLocalDB/Pluralia?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
)

print(f"Using DATABASE_URL: {DATABASE_URL.split('@')[0]}@[HIDDEN]")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def fix_stuck_job():
    """Manually fix stuck job by creating paragraphs and updating status"""

    print("="*80)
    print(f"FIXING STUCK JOB: {JOB_ID}")
    print("="*80)

    db = SessionLocal()
    try:
        # 1. Check job status
        result = db.execute(text("""
            SELECT status, original_filename, is_post_processed
            FROM jobs
            WHERE id = :job_id
        """), {"job_id": JOB_ID})

        job = result.fetchone()
        if not job:
            print(f"ERROR: Job {JOB_ID} not found!")
            return

        print(f"\n1. Current Job Status:")
        print(f"   - Status: {job[0]}")
        print(f"   - Filename: {job[1]}")
        print(f"   - Is post processed: {job[2]}")

        # 2. Check existing paragraphs
        result = db.execute(text("""
            SELECT COUNT(*) FROM paragraphs WHERE job_id = :job_id
        """), {"job_id": JOB_ID})

        existing_paragraphs = result.scalar()
        print(f"\n2. Existing paragraphs: {existing_paragraphs}")

        if existing_paragraphs > 0:
            print("   Paragraphs already exist. Updating status only...")
            db.execute(text("""
                UPDATE jobs
                SET status = 'COMPLETED', is_post_processed = 1
                WHERE id = :job_id
            """), {"job_id": JOB_ID})
            db.commit()
            print("   Status updated to COMPLETED")
            return

        # 3. Get grouped turns from stored procedure
        print("\n3. Getting grouped turns from stored procedure...")
        result = db.execute(text(f"""
            SELECT * FROM dbo.GetGroupedTurnsByJob('{JOB_ID}')
            ORDER BY turn_id
        """))

        turns = result.fetchall()
        print(f"   Found {len(turns)} grouped turns")

        if len(turns) == 0:
            print("   ERROR: No grouped turns found!")
            return

        # 4. Insert paragraphs
        print("\n4. Creating paragraphs...")
        for turn in turns:
            db.execute(text("""
                INSERT INTO paragraphs (
                    id, job_id, start_timestamp, end_timestamp,
                    speaker, original_text, formatted_text,
                    language, status, confidence
                )
                VALUES (
                    NEWID(), :job_id, :start_timestamp, :end_timestamp,
                    :speaker, :original_text, :formatted_text,
                    :language, 'pending', NULL
                )
            """), {
                "job_id": JOB_ID,
                "start_timestamp": str(turn.turn_start),
                "end_timestamp": str(turn.turn_end),
                "speaker": turn.speaker,
                "original_text": turn.combined_text,
                "formatted_text": turn.combined_text,
                "language": turn.primary_language
            })

        print(f"   Inserted {len(turns)} paragraphs")

        # 5. Mark diarization segments as processed
        print("\n5. Marking diarization segments as processed...")
        result = db.execute(text("""
            UPDATE diarization_segments
            SET status = 'PROCESSED'
            WHERE job_id = :job_id AND status != 'PROCESSED'
        """), {"job_id": JOB_ID})
        print(f"   Updated {result.rowcount} diarization segments")

        # 6. Update job status
        print("\n6. Updating job status to COMPLETED...")
        db.execute(text("""
            UPDATE jobs
            SET status = 'COMPLETED', is_post_processed = 1
            WHERE id = :job_id
        """), {"job_id": JOB_ID})

        db.commit()

        print("\n" + "="*80)
        print("SUCCESS! Job fixed successfully")
        print("="*80)
        print(f"  - Total paragraphs created: {len(turns)}")
        print(f"  - Status: COMPLETED")
        print(f"  - Is post processed: True")
        print("="*80)

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    fix_stuck_job()