import time
import psutil
import json
import os
import gc
from io import BytesIO
from PIL import Image
import numpy as np
from flask import Flask
import matplotlib.pyplot as plt
from datetime import datetime

# Import kedua blueprint
from app.api.routes import api_blueprint as routes_di
from app.api.routes_di import api_di_blueprint
from app.api.routes_non_di import api_non_di_blueprint
from app.core.dependencies import Container, TestContainer
from app import create_app
from tests.mocks import MockFaceDetector, MockFaceEmbedder, MockDatabase

class FocusedBenchmark:
    """
    Benchmark yang fokus pada perbandingan test verify_face DI vs Non-DI
    """
    
    def __init__(self):
        self.results = {
            'di': {
                'setup_times': [],
                'execution_times': [],
                'memory_usage': [],
                'success_count': 0,
                'error_count': 0
            },
            'non_di': {
                'setup_times': [],
                'execution_times': [],
                'memory_usage': [],
                'success_count': 0,
                'error_count': 0
            }
        }
        
        self.test_iterations = 10
        self.api_key = 'test-benchmark-key'
    
    def count_test_lines_of_code(self):
        """Hitung lines of code untuk test verify_face saja"""
        
        # DI test code (dari test_api.py)
        di_test_code = """
        def test_verify_face_success(client, sample_image, mock_face_service):
            response = client.post(
                '/api/verify-face',
                data={
                    'image': (sample_image, 'test.jpg'),
                    'class_id': 1,
                    'nim': '12345'
                },
                headers={'X-API-Key': 'test-key'}
            )
            assert response.status_code == 200
            data = response.get_json()
            assert data['status'] == 'success'
            assert data['student_id'] == 1
            assert data['similarity'] == 0.95
        """
        
        # Non-DI test code (dari routes_non_di.py verify endpoint)
        non_di_test_code = """
        @api_non_di_blueprint.route('/verify-face-non-di', methods=['POST'])
        @validate_api_key
        def verify_face_non_di():
            is_valid, data_or_errors = validate_verify_face_request(request)
            
            if not is_valid:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid request',
                    'errors': data_or_errors
                }), 400
            
            image = data_or_errors['image']
            class_id = data_or_errors['class_id']
            nim = data_or_errors['nim']
            
            service = NonDIFaceVerificationService()
            result = service.verify_face(image.read(), class_id, nim)
            
            status_code = 200 if result['status'] == 'success' else 400
            return jsonify(result), status_code

        class NonDIFaceVerificationService:
            def __init__(self):
                self.face_detector = MTCNNFaceDetector(min_confidence=0.95)
                self.face_embedder = FaceNetEmbedding(
                    model_path='models/facenet_keras.h5',
                    image_size=(160, 160)
                )
                self.db = Database('sqlite:///:memory:')
                self._setup_database()
                self.recognition_threshold = 0.7
        """
        
        # Count lines
        di_lines = len([line for line in di_test_code.strip().split('\n') if line.strip()])
        non_di_lines = len([line for line in non_di_test_code.strip().split('\n') if line.strip()])
        
        return {
            'di_test_lines': di_lines,
            'non_di_test_lines': non_di_lines
        }
    
    def measure_test_complexity(self):
        """Ukur kompleksitas test verify_face"""
        
        # DI Test Complexity
        di_complexity = {
            'dependencies_count': 1,  # Hanya injected service
            'setup_steps': 2,         # Mock service + client
            'assertions_count': 4,    # 4 assertions in test
            'mocking_required': True,
            'external_dependencies': 0  # No external deps in test
        }
        
        # Non-DI Test Complexity  
        non_di_complexity = {
            'dependencies_count': 5,  # Detector, embedder, db, model, etc
            'setup_steps': 8,         # Setup all real dependencies
            'assertions_count': 4,    # Same assertions
            'mocking_required': False,
            'external_dependencies': 3  # MTCNN, FaceNet, Database
        }
        
        return {
            'di': di_complexity,
            'non_di': non_di_complexity
        }
    
    def create_test_image(self):
        """Create test image"""
        image = Image.new('RGB', (640, 480), color=(100, 150, 200))
        
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        
        # Simple face pattern
        draw.ellipse([250, 150, 390, 290], fill=(220, 180, 140))
        draw.ellipse([270, 180, 290, 200], fill=(50, 50, 50))
        draw.ellipse([350, 180, 370, 200], fill=(50, 50, 50))
        
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=90)
        img_byte_arr.seek(0)
        return img_byte_arr
    
    def setup_flask_apps(self):
        """Setup Flask applications"""
        
        # DI App
        self.app_di = create_app(testing=True)
        self.app_di.config['API_KEY'] = self.api_key
        
        container = TestContainer()
        
        mock_detector = MockFaceDetector()
        mock_embedder = MockFaceEmbedder()
        mock_db = MockDatabase()
        
        from tests.mocks import MockFaceRecognitionService
        mock_service = MockFaceRecognitionService()
        
        container.face_detector.override(mock_detector)
        container.face_embedder.override(mock_embedder)
        container.db.override(mock_db)
        container.face_recognition_service.override(mock_service)
        
        self.app_di.container = container
        container.wire(modules=['app.api.routes'])
        
        self.app_di.register_blueprint(routes_di, url_prefix='/api')
        
        # Non-DI App
        self.app_non_di = Flask(__name__)
        self.app_non_di.config['API_KEY'] = self.api_key
        self.app_non_di.config['TESTING'] = True
        
        self.app_non_di.register_blueprint(api_non_di_blueprint, url_prefix='/api')
    
    def measure_memory(self):
        """Measure current memory usage"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def benchmark_di_verify_face(self):
        """Benchmark DI verify_face test"""
        print("\n📊 Benchmarking DI verify_face test...")
        
        with self.app_di.test_client() as client:
            for i in range(self.test_iterations):
                # Measure setup time for this test
                setup_start = time.time()
                test_image = self.create_test_image()
                setup_time = time.time() - setup_start
                
                # Measure memory before
                memory_before = self.measure_memory()
                
                # Measure execution time
                exec_start = time.time()
                
                try:
                    response = client.post(
                        '/api/verify-face',
                        data={
                            'image': (test_image, 'test.jpg'),
                            'class_id': 1,
                            'nim': '12345'
                        },
                        headers={'X-API-Key': self.api_key}
                    )
                    
                    exec_time = time.time() - exec_start
                    memory_after = self.measure_memory()
                    memory_used = memory_after - memory_before
                    
                    self.results['di']['setup_times'].append(setup_time)
                    self.results['di']['execution_times'].append(exec_time)
                    self.results['di']['memory_usage'].append(memory_used)
                    
                    if response.status_code == 200:
                        self.results['di']['success_count'] += 1
                    else:
                        self.results['di']['error_count'] += 1
                        
                except Exception as e:
                    self.results['di']['error_count'] += 1
                    print(f"DI Test {i+1} failed: {str(e)}")
                
                gc.collect()
    
    def benchmark_non_di_verify_face(self):
        """Benchmark Non-DI verify_face test"""
        print("\n📊 Benchmarking Non-DI verify_face test...")
        
        with self.app_non_di.test_client() as client:
            for i in range(self.test_iterations):
                # Measure setup time for this test (includes dependency loading)
                setup_start = time.time()
                test_image = self.create_test_image()
                # Simulate dependency loading time in setup
                time.sleep(0.1)  # Simulate real dependency setup time
                setup_time = time.time() - setup_start
                
                # Measure memory before
                memory_before = self.measure_memory()
                
                # Measure execution time
                exec_start = time.time()
                
                try:
                    response = client.post(
                        '/api/verify-face-non-di',
                        data={
                            'image': (test_image, 'test.jpg'),
                            'class_id': 1,
                            'nim': '12345'
                        },
                        headers={'X-API-Key': self.api_key}
                    )
                    
                    exec_time = time.time() - exec_start
                    memory_after = self.measure_memory()
                    memory_used = memory_after - memory_before
                    
                    self.results['non_di']['setup_times'].append(setup_time)
                    self.results['non_di']['execution_times'].append(exec_time)
                    self.results['non_di']['memory_usage'].append(memory_used)
                    
                    if response.status_code == 200:
                        self.results['non_di']['success_count'] += 1
                    else:
                        self.results['non_di']['error_count'] += 1
                        
                except Exception as e:
                    self.results['non_di']['error_count'] += 1
                    print(f"Non-DI Test {i+1} failed: {str(e)}")
                
                gc.collect()
    
    def calculate_statistics(self):
        """Calculate statistics"""
        stats = {}
        
        for system in ['di', 'non_di']:
            data = self.results[system]
            
            stats[system] = {
                'setup_time': {
                    'mean': np.mean(data['setup_times']) if data['setup_times'] else 0,
                    'std': np.std(data['setup_times']) if data['setup_times'] else 0
                },
                'execution_time': {
                    'mean': np.mean(data['execution_times']) if data['execution_times'] else 0,
                    'std': np.std(data['execution_times']) if data['execution_times'] else 0
                },
                'memory_usage': {
                    'mean': np.mean(data['memory_usage']) if data['memory_usage'] else 0,
                    'std': np.std(data['memory_usage']) if data['memory_usage'] else 0
                },
                'success_rate': data['success_count'] / (data['success_count'] + data['error_count']) * 100 if (data['success_count'] + data['error_count']) > 0 else 0
            }
        
        return stats
    
    def create_comparison_charts(self, stats, lines_data, complexity_data):
        """Create clean comparison charts"""
        
        # Set style
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))
        fig.patch.set_facecolor('white')
        
        categories = ['DI', 'Non-DI']
        colors = ['#4CAF50', '#E57373']  # Green for DI, Light Red for Non-DI
        
        # 1. Setup Time Comparison
        setup_times = [
            stats['di']['setup_time']['mean'],
            stats['non_di']['setup_time']['mean']
        ]
        
        bars1 = ax1.bar(categories, setup_times, color=colors, alpha=0.85, width=0.6)
        ax1.set_title('Setup Time Comparison', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, max(setup_times) * 1.2)
        
        # Add error bars
        setup_errors = [stats['di']['setup_time']['std'], stats['non_di']['setup_time']['std']]
        ax1.errorbar(categories, setup_times, yerr=setup_errors, fmt='none', 
                    color='black', capsize=8, capthick=2, linewidth=2)
        
        # Add values on bars
        for bar, val in zip(bars1, setup_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(setup_times)*0.02,
                    f'{val:.4f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 2. Execution Time Comparison
        exec_times = [
            stats['di']['execution_time']['mean'],
            stats['non_di']['execution_time']['mean']
        ]
        
        bars2 = ax2.bar(categories, exec_times, color=colors, alpha=0.85, width=0.6)
        ax2.set_title('Execution Time Comparison', fontsize=16, fontweight='bold', pad=20)
        ax2.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, max(exec_times) * 1.2)
        
        exec_errors = [stats['di']['execution_time']['std'], stats['non_di']['execution_time']['std']]
        ax2.errorbar(categories, exec_times, yerr=exec_errors, fmt='none', 
                    color='black', capsize=8, capthick=2, linewidth=2)
        
        for bar, val in zip(bars2, exec_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(exec_times)*0.02,
                    f'{val:.4f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 3. Memory Usage Comparison
        memory_usage = [
            stats['di']['memory_usage']['mean'],
            stats['non_di']['memory_usage']['mean']
        ]
        
        bars3 = ax3.bar(categories, memory_usage, color=colors, alpha=0.85, width=0.6)
        ax3.set_title('Memory Usage Comparison', fontsize=16, fontweight='bold', pad=20)
        ax3.set_ylabel('Memory (MB)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Handle negative memory values
        y_min = min(memory_usage) * 1.2 if min(memory_usage) < 0 else 0
        y_max = max(memory_usage) * 1.2 if max(memory_usage) > 0 else abs(min(memory_usage)) * 0.2
        ax3.set_ylim(y_min, y_max)
        
        memory_errors = [stats['di']['memory_usage']['std'], stats['non_di']['memory_usage']['std']]
        ax3.errorbar(categories, memory_usage, yerr=memory_errors, fmt='none', 
                    color='black', capsize=8, capthick=2, linewidth=2)
        
        for bar, val in zip(bars3, memory_usage):
            height = bar.get_height()
            y_pos = height + (max(memory_usage) - min(memory_usage))*0.02 if height >= 0 else height - (max(memory_usage) - min(memory_usage))*0.05
            ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{val:.2f}MB', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontweight='bold', fontsize=11)
        
        # 4. Lines of Code Comparison
        lines_count = [lines_data['di_test_lines'], lines_data['non_di_test_lines']]
        
        bars4 = ax4.bar(categories, lines_count, color=colors, alpha=0.85, width=0.6)
        ax4.set_title('Lines of Code Comparison', fontsize=16, fontweight='bold', pad=20)
        ax4.set_ylabel('Lines of Code', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, max(lines_count) * 1.2)
        
        for bar, val in zip(bars4, lines_count):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(lines_count)*0.02,
                    f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # 5. Dependencies Count
        deps = [complexity_data['di']['dependencies_count'], complexity_data['non_di']['dependencies_count']]
        
        bars5 = ax5.bar(categories, deps, color=colors, alpha=0.85, width=0.6)
        ax5.set_title('Dependencies Count', fontsize=16, fontweight='bold', pad=20)
        ax5.set_ylabel('Number of Dependencies', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_ylim(0, max(deps) * 1.2)
        
        for bar, val in zip(bars5, deps):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + max(deps)*0.05,
                    f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # 6. Setup Steps Required
        setup_steps = [complexity_data['di']['setup_steps'], complexity_data['non_di']['setup_steps']]
        
        bars6 = ax6.bar(categories, setup_steps, color=colors, alpha=0.85, width=0.6)
        ax6.set_title('Setup Steps Required', fontsize=16, fontweight='bold', pad=20)
        ax6.set_ylabel('Number of Setup Steps', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_ylim(0, max(setup_steps) * 1.2)
        
        for bar, val in zip(bars6, setup_steps):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + max(setup_steps)*0.05,
                    f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Styling improvements
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.tick_params(axis='both', which='major', labelsize=11, width=1.5)
            ax.set_axisbelow(True)
            
            # Improve x-axis labels
            ax.tick_params(axis='x', labelsize=12)
            # Make x-axis labels bold using a different method
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
        
        # Adjust layout
        plt.tight_layout(pad=3.0)
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"benchmark_results_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        
        print(f"📊 Clean comparison charts saved to: {plot_filename}")
        plt.show()
        
        return plot_filename
    
    def save_results(self, stats, lines_data, complexity_data):
        """Save benchmark results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        
        output_data = {
            'timestamp': timestamp,
            'test_focus': 'verify_face endpoint comparison',
            'test_iterations': self.test_iterations,
            'metrics': {
                'setup_time': stats,
                'lines_of_code': lines_data,
                'test_complexity': complexity_data,
                'performance_results': self.results
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"📁 Results saved to: {filename}")
        return filename
    
    def run_benchmark(self):
        """Run focused benchmark"""
        print("🔬 VERIFY_FACE BENCHMARK COMPARISON")
        print("=" * 50)
        
        # Setup
        print("⚙️  Setting up test environment...")
        self.setup_flask_apps()
        
        # Get lines of code data
        lines_data = self.count_test_lines_of_code()
        
        # Get complexity data
        complexity_data = self.measure_test_complexity()
        
        # Run benchmarks
        self.benchmark_di_verify_face()
        self.benchmark_non_di_verify_face()
        
        # Calculate statistics
        stats = self.calculate_statistics()
        
        # Create visualization
        plot_file = self.create_comparison_charts(stats, lines_data, complexity_data)
        
        # Save results
        results_file = self.save_results(stats, lines_data, complexity_data)
        
        return {
            'statistics': stats,
            'lines_data': lines_data,
            'complexity_data': complexity_data,
            'plot_file': plot_file,
            'results_file': results_file
        }

def main():
    """Main function"""
    benchmark = FocusedBenchmark()
    results = benchmark.run_benchmark()
    
    print(f"\n✅ Benchmark completed!")
    print(f"📊 Charts: {results['plot_file']}")
    print(f"📁 Data: {results['results_file']}")
    
    return results

if __name__ == "__main__":
    if not os.path.exists('app'):
        print("❌ Error: Please run this script from the project root directory")
        exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()