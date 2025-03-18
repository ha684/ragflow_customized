import os
import sys
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            '../../')))

from deepdoc.vision.seeit import draw_box
from deepdoc.vision import OCR, init_in_out
import argparse
import numpy as np
import trio
import aiohttp
import json
import asyncio
from PIL import Image
import io
from pathlib import Path
import ssl

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2' #2 gpus, uncontinuous
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' #1 gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '' #cpu

# Create a context for asyncio to run in trio
class TrioAsyncioContext:
    def __init__(self):
        self.loop = None

    async def __aenter__(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        return self.loop

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.loop.close()
        asyncio.set_event_loop(None)

async def check_endpoint_availability(host, port, trio_token=None):
    """Check if the vision API endpoint is available."""
    url = f"{host}:{port}/vision/health" if port else f"{host}/vision/health"
    
    # Fix URL format if needed
    if not url.startswith(('http://', 'https://')):
        url = f"https://{url}" if 'ngrok' in url else f"http://{url}"
    
    print(f"Checking endpoint at: {url}")
    
    async with TrioAsyncioContext() as loop:
        result = await trio.to_thread.run_sync(
            lambda _: asyncio.run(async_check_endpoint(url)), 
            trio_token
        )
        return result

async def async_check_endpoint(url):
    """Run in asyncio context to check endpoint."""
    try:
        # Create a ClientSession with SSL verification disabled
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            print(f"Checking endpoint: {url}")
            async with session.get(url, timeout=10) as response:
                status = response.status
                print(f"Endpoint response: status={status}")
                return status == 200
    except Exception as e:
        print(f"Error checking endpoint: {repr(e)}")
        return False

async def process_with_endpoint(img_path, query, host, port, trio_token=None):
    """Process the image using the remote vision API endpoint."""
    url = f"{host}:{port}/vision/vision" if port else f"{host}/vision/vision"
    
    # Fix URL format if needed
    if not url.startswith(('http://', 'https://')):
        url = f"https://{url}" if 'ngrok' in url else f"http://{url}"
    
    async with TrioAsyncioContext() as loop:
        result = await trio.to_thread.run_sync(
            lambda _: asyncio.run(async_process_endpoint(url, img_path, query)), 
            trio_token
        )
        return result

async def async_process_endpoint(url, img_path, query):
    """Run in asyncio context to process with endpoint."""
    # Open and prepare the image file
    try:
        # Create a ClientSession with SSL verification disabled
        connector = aiohttp.TCPConnector(ssl=False)
        
        with open(img_path, 'rb') as img_file:
            form_data = aiohttp.FormData()
            form_data.add_field('file', 
                              img_file, 
                              filename=os.path.basename(img_path),
                              content_type='image/jpeg')  # Adjust content type if needed
            form_data.add_field('query', query)
            
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(url, data=form_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        print(f"Error from vision API: {response.status} - {error_text}")
                        return None
    except Exception as e:
        print(f"Error connecting to vision API: {str(e)}")
        return None

def main(args):
    import torch.cuda
    
    cuda_devices = torch.cuda.device_count()
    limiter = [trio.CapacityLimiter(1) for _ in range(cuda_devices)] if cuda_devices > 1 else None
    ocr = OCR()
    images, outputs = init_in_out(args)
    
    def __ocr(i, id, img):
        print(f"Task {i} start with local OCR")
        bxs = ocr(np.array(img), id)
        bxs = [(line[0], line[1][0]) for line in bxs]
        bxs = [{
            "text": t,
            "bbox": [b[0][0], b[0][1], b[1][0], b[-1][1]],
            "type": "ocr",
            "score": 1} for b, t in bxs if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]]
        img = draw_box(images[i], bxs, ["ocr"], 1.)
        img.save(outputs[i], quality=95)
        with open(outputs[i] + ".txt", "w+", encoding='utf-8') as f:
            f.write("\n".join([o["text"] for o in bxs]))
        
        print(f"Task {i} done")
    
    async def __ocr_thread(i, id, img, limiter=None):
        if limiter:
            async with limiter:
                print(f"Task {i} use device {id}")
                await trio.to_thread.run_sync(lambda: __ocr(i, id, img))
        else:
            await trio.to_thread.run_sync(lambda: __ocr(i, id, img))
    
    async def __process_with_api(i, img_path, img, trio_token):
        try:
            print(f"Task {i} start with Vision API")
            # Save the image temporarily if it's not already a file
            if isinstance(img_path, Image.Image):
                temp_img_path = f"temp_img_{i}.jpg"
                img.save(temp_img_path, quality=95)
                img_path = temp_img_path
            
            # Process using the Vision API
            result = await process_with_endpoint(
                img_path, 
                args.vision_query,
                args.endpoint_host,
                args.endpoint_port,
                trio_token
            )
            
            if result and "result" in result:
                # Assuming the API returns OCR results in a format we can parse
                text_content = result["result"]
                
                # Save the extracted text to a file
                with open(outputs[i] + ".txt", "w+", encoding='utf-8') as f:
                    f.write(text_content)
                
                print(f"Task {i} done with Vision API")
                return True
            else:
                print(f"Task {i} failed with Vision API, falling back to local OCR")
                return False
        except Exception as e:
            print(f"Error in Vision API processing for task {i}: {str(e)}")
            print(f"Falling back to local OCR")
            return False
        finally:
            # Clean up temporary file if created
            if isinstance(img_path, str) and img_path.startswith("temp_img_"):
                try:
                    os.remove(img_path)
                except:
                    pass
    
    async def __ocr_launcher():
        # Create a trio token for thread sync
        trio_token = trio.lowlevel.current_trio_token()
        
        # Check if the endpoint is available if endpoint options are provided
        use_endpoint = False
        if args.endpoint_host:
            print(f"Checking Vision API endpoint...")
            use_endpoint = await check_endpoint_availability(args.endpoint_host, args.endpoint_port, trio_token)
            if use_endpoint:
                print(f"Vision API endpoint is available at {args.endpoint_host}:{args.endpoint_port if args.endpoint_port else ''}")
            else:
                print(f"Vision API endpoint is not available, falling back to local OCR")
        
        # If --use_both flag is provided, process both methods
        if args.use_both and use_endpoint:
            print("Using both Vision API and local OCR as requested")
            
            for i, img in enumerate(images):
                # Process with Vision API
                img_path = args.inputs if len(images) == 1 else None
                if isinstance(img_path, str) and os.path.isfile(img_path):
                    # Vision API processing
                    api_success = await __process_with_api(i, img_path, img, trio_token)
                    
                    # Store API results in a separate location
                    if api_success:
                        # Save with a different name to differentiate from local OCR
                        api_output = outputs[i].replace('.jpg', '_api.jpg')
                        api_txt = outputs[i].replace('.jpg', '_api.txt') 
                        try:
                            import shutil
                            shutil.copy(outputs[i], api_output)
                            shutil.copy(outputs[i] + ".txt", api_txt)
                            print(f"Saved Vision API results for task {i} to {api_output}")
                        except Exception as e:
                            print(f"Error saving API results: {str(e)}")
                
                # Always process with local OCR when --use_both is specified
                if cuda_devices > 1:
                    await __ocr_thread(i, i % cuda_devices, img, limiter[i % cuda_devices])
                else:
                    await __ocr_thread(i, 0, img)
        
        elif use_endpoint:
            # Process using only the Vision API (with fallback to local OCR on failure)
            for i, img in enumerate(images):
                img_path = args.inputs if len(images) == 1 else None
                if isinstance(img_path, str) and os.path.isfile(img_path):
                    success = await __process_with_api(i, img_path, img, trio_token)
                    if not success:
                        # Fall back to local OCR if API processing fails
                        if cuda_devices > 1:
                            await __ocr_thread(i, i % cuda_devices, img, limiter[i % cuda_devices])
                        else:
                            await __ocr_thread(i, 0, img)
                else:
                    print(f"Image path is not a valid file: {img_path}")
                    await __ocr_thread(i, 0, img)
        else:
            # Use local OCR processing
            if cuda_devices > 1:
                async with trio.open_nursery() as nursery:
                    for i, img in enumerate(images):
                        nursery.start_soon(__ocr_thread, i, i % cuda_devices, img, limiter[i % cuda_devices])
                        await trio.sleep(0.1)
            else:
                for i, img in enumerate(images):
                    await __ocr_thread(i, 0, img)
    
    trio.run(__ocr_launcher)
    
    print("OCR tasks are all done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',
                        help="Directory where to store images or PDFs, or a file path to a single image or PDF",
                        required=True)
    parser.add_argument('--output_dir', help="Directory where to store the output images. Default: './ocr_outputs'",
                        default="./ocr_outputs")
    parser.add_argument('--endpoint_host', help="Host for the Vision API endpoint. Default: 'localhost'",
                        default="localhost")
    parser.add_argument('--endpoint_port', help="Port for the Vision API endpoint. Default: 8000",
                        default=None)
    parser.add_argument('--use_api', help="Force using the Vision API (if available)", action="store_true")
    parser.add_argument('--use_both', help="Use both Vision API and local OCR", action="store_true")
    parser.add_argument('--vision_query', help="Query to send to the Vision API. Default is to extract all text.",
                        default="OCR")
    args = parser.parse_args()
    main(args)