# Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2' #2 gpus, uncontinuous
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #1 gpu
# os.environ['CUDA_VISIBLE_DEVICES'] = '' #cpu

async def check_endpoint_availability(host, port):
    """Check if the vision API endpoint is available."""
    url = f"http://{host}:{port}/vision/health" if port else f"http://{host}/vision/health"  # Assuming there's a health endpoint
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=2) as response:
                return response.status == 200
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return False

async def process_with_endpoint(img_path, query, host, port):
    """Process the image using the remote vision API endpoint."""
    url = f"http://{host}:{port}/vision/vision" if port else f"http://{host}/vision/vision"
    
    # Open and prepare the image file
    with open(img_path, 'rb') as img_file:
        form_data = aiohttp.FormData()
        form_data.add_field('file', 
                           img_file, 
                           filename=os.path.basename(img_path),
                           content_type='image/jpeg')  # Adjust content type if needed
        form_data.add_field('query', query)
        
        try:
            async with aiohttp.ClientSession() as session:
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
            __ocr(i, id, img)
    
    async def __process_with_api(i, img_path, img):
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
                args.endpoint_port
            )
            
            if result and "result" in result:
                # Assuming the API returns OCR results in a format we can parse
                # You may need to adjust this based on your API's actual response format
                text_content = result["result"]
                
                # Save the original image with annotations if available
                img = draw_box(images[i], [], ["ocr"], 1.)  # No boxes from API
                img.save(outputs[i], quality=95)
                
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
        # Check if the endpoint is available if endpoint options are provided
        use_endpoint = False
        if args.endpoint_host and args.endpoint_port:
            use_endpoint = await check_endpoint_availability(args.endpoint_host, args.endpoint_port)
            if use_endpoint:
                print(f"Vision API endpoint is available at {args.endpoint_host}:{args.endpoint_port}")
            else:
                print(f"Vision API endpoint is not available, falling back to local OCR")
        
        if use_endpoint:
            # Process using the Vision API
            for i, img in enumerate(images):
                img_path = args.inputs if len(images) == 1 else None
                success = await __process_with_api(i, img_path, img)
                if not success:
                    # Fall back to local OCR if API processing fails
                    if cuda_devices > 1:
                        async with limiter[i % cuda_devices]:
                            await trio.to_thread.run_sync(lambda: __ocr(i, i % cuda_devices, img))
                    else:
                        await trio.to_thread.run_sync(lambda: __ocr(i, 0, img))
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
    parser.add_argument('--vision_query', help="Query to send to the Vision API. Default is to extract all text.",
                        default="OCR")
    args = parser.parse_args()
    main(args)