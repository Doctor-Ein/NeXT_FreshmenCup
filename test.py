import asyncio
from AWS_Service.Transcribe import TranscribeService

async def main():
    # Create event loop and service instance
    loop = asyncio.get_event_loop()
    transcribe_service = TranscribeService(loop)

    try:
        # Start the transcription service
        print("Starting transcription service...")
        await transcribe_service.start()

        # Run for some time while processing transcripts
        print("Listening for speech (say something)...")
        for _ in range(30):  # Run for about 30 seconds
            await asyncio.sleep(1)
            
            # Get available transcripts
            transcript = await transcribe_service.get_transcript()
            if transcript:
                print(f"Transcript: {transcript}")

            # Demonstrate pause/resume functionality
            if _ == 10:
                print("\nPausing transcription for 5 seconds...")
                transcribe_service.pause()
            elif _ == 15:
                print("Resuming transcription...")
                transcribe_service.resume()

    except KeyboardInterrupt:
        print("\nUser interrupted...")
    finally:
        # Clean up
        print("Stopping transcription service...")
        await transcribe_service.stop()

if __name__ == "__main__":
    asyncio.run(main())